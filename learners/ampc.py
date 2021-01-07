#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import gym
import numpy as np
from gym.envs.user_defined.toyota_exp_cbf.dynamics_and_models import EnvironmentModel

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf
    tf.config.optimizer.set_experimental_options({'constant_folding': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  })

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EnvironmentModel(**args2envkwargs(args))
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.model.args = self.args
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf

    def barrier_lambda_schedule(self, ite):
        init_lambda = self.args.barrier_lambda_scheduler[0]
        end_lambda = self.args.barrier_lambda_scheduler[2]
        end_ite = self.args.barrier_lambda_scheduler[1]
        interval = self.args.barrier_lambda_interval
        if ite <= end_ite:
            factor =  self.tf.cast(ite//interval, self.tf.float32) / self.tf.cast(end_ite//interval, self.tf.float32)
            barrier_lambda = (init_lambda - end_lambda) *(1 - factor)  + end_lambda
        else:
            barrier_lambda = self.tf.convert_to_tensor(end_lambda)
        return barrier_lambda

    def model_rollout_for_update(self, start_obses, ite, mb_ref_index):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses, mb_ref_index)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        punish_terms_for_training_sum = self.tf.zeros((start_obses.shape[0],))
        veh2veh4training_sum = self.tf.zeros((start_obses.shape[0],))
        veh2veh4real_sum = self.tf.zeros((start_obses.shape[0],))
        veh2road4real_sum = self.tf.zeros((start_obses.shape[0],))
        veh2veh41step_sum = self.tf.zeros((start_obses.shape[0],))
        veh2veh4init_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        pf = self.punish_factor_schedule(ite)
        self.model.barrier_lambda = self.barrier_lambda_schedule(ite) #todo: model and labmda intereaction
        for step in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses) # preprocess: normalize, scale
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, rewards, veh2veh4training, veh2road4training, veh2veh4real, veh2road4real = self.model.rollout_out(actions)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            veh2veh4real_sum += veh2veh4real
            veh2road4real_sum += veh2road4real
            punish_terms_for_training_sum += veh2road4training
            if step == 0:
                veh2veh4init_sum += veh2veh4real
            if step == 1:
                veh2veh41step_sum += veh2veh4real
            if step >= 1 and step <= self.args.barrier_steps:
                veh2veh4training_sum += veh2veh4training

        # loss
        obj_loss = -self.tf.reduce_mean(rewards_sum) #
        punish_terms_for_training_sum += veh2veh4training_sum
        veh2veh4training = self.tf.reduce_mean(veh2veh4training_sum)
        punish_term_for_training = self.tf.reduce_mean(punish_terms_for_training_sum)
        veh2veh4real = self.tf.reduce_mean(veh2veh4real_sum)
        veh2road4real = self.tf.reduce_mean(veh2road4real_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term_for_training
        total_loss = obj_loss + punish_loss

        # dist
        veh2veh4init = self.tf.reduce_mean(veh2veh4init_sum)
        veh2veh41step = self.tf.reduce_mean(veh2veh41step_sum)

        return obj_loss, punish_term_for_training, punish_loss, total_loss, pf, veh2veh4real, veh2road4real, veh2veh4training, veh2veh4init, veh2veh41step

    @tf.function
    def forward_and_backward(self, mb_obs, ite, mb_ref_index):
        with self.tf.GradientTape() as tape:
            obj_loss, punish_term_for_training, punish_loss, total_loss, \
            pf, veh2veh4real, veh2road4real, veh2veh4training, \
            veh2veh4init, veh2veh41step = self.model_rollout_for_update(mb_obs, ite, mb_ref_index)

        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(total_loss, self.policy_with_value.policy.trainable_weights)

        return pg_grad, obj_loss, \
               punish_term_for_training, punish_loss, total_loss,\
               veh2road4real, veh2veh4real, veh2veh4training, pf, veh2veh4init, veh2veh41step

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            pg_grad, obj_loss, \
            punish_term_for_training, punish_loss, total_loss, \
            veh2road4real, veh2veh4real, veh2veh4training, pf, veh2veh4init, veh2veh41step =\
                self.forward_and_backward(mb_obs, iteration, mb_ref_index)

            pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term_for_training=punish_term_for_training.numpy(),
            veh2veh4real=veh2veh4real.numpy(),
            veh2road4real=veh2road4real.numpy(),
            veh2veh4init=veh2veh4init.numpy(),
            veh2veh41step=veh2veh41step.numpy(),
            veh2veh4training=veh2veh4training.numpy(),
            punish_loss=punish_loss.numpy(),
            pg_loss=total_loss.numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            barrier_lambda=self.barrier_lambda_schedule(iteration).numpy()
        ))

        grads = pg_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
