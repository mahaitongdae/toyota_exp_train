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
from gym.envs.vehicle.toyota_exp.dynamics_and_models import EnvironmentModel

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
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        # self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
        #                    'batch_actions': batch_data[1].astype(np.float32),
        #                    'batch_rewards': batch_data[2].astype(np.float32),
        #                    'batch_obs_tp1': batch_data[3].astype(np.float32),
        #                    'batch_dones': batch_data[4].astype(np.float32),
        #                    'batch_ref_index': batch_data[5].astype(np.int32),
        #                    'batch_veh_num': batch_data[6],
        #                    'batch_veh_mode': batch_data[7],
        #                    }
        # print(batch_data[0])
        self.batch_data = {'batch_obs': batch_data[0],
                           'batch_actions': batch_data[1],
                           'batch_rewards': batch_data[2],
                           'batch_obs_tp1': batch_data[3],
                           'batch_dones': batch_data[4],
                           'batch_ref_index': batch_data[5],
                           'batch_veh_num': batch_data[6],
                           'batch_veh_mode': batch_data[7],
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
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite // interval, self.tf.float32))
        return pf

    def model_rollout_for_update(self, start_obs, ite, ref_index, veh_num, veh_mode):
        start_obs = self.tf.tile(start_obs, [self.M, 1])
        self.model.reset(start_obs, ref_index, veh_num, veh_mode)
        rewards_sum = self.tf.zeros((start_obs.shape[0],))
        punish_terms_for_training_sum = self.tf.zeros((start_obs.shape[0],))
        real_punish_terms_sum = self.tf.zeros((start_obs.shape[0],))
        veh2veh4real_sum = self.tf.zeros((start_obs.shape[0],))
        veh2road4real_sum = self.tf.zeros((start_obs.shape[0],))
        obs = start_obs
        pf = self.punish_factor_schedule(ite)

        for i in range(self.num_rollout_list_for_policy_update[0]):
            # print('rollout step', i)
            obs_ego_scale = self.tf.convert_to_tensor([[0.2, 1., 2., 1 / 30., 1 / 30, 1 / 180.]], dtype=self.tf.float32)
            obs_track_scale = self.tf.convert_to_tensor(
                [[1., 1 / 15., 0.2] + [1., 1., 1 / 15.] * self.args.env_kwargs_num_future_data], dtype=self.tf.float32)
            obs_other_scale = self.tf.tile([[1 / 30., 1 / 30., 0.2, 1 / 180.]], [1, int(
                (obs.shape[1] - self.args.state_ego_dim - self.args.state_track_dim) / self.args.state_other_dim)])
            obs_scale = self.tf.concat([obs_ego_scale, obs_track_scale, obs_other_scale], axis=1)

            self.preprocessor.obs_scale = obs_scale

            processed_obs = self.preprocessor.tf_process_obses(obs)
            # print('processed_obs', processed_obs)
            obs_ego, obs_other = processed_obs[:, 0: self.args.state_ego_dim + self.args.state_track_dim], \
                                 processed_obs[:, self.args.state_ego_dim + self.args.state_track_dim:]

            obs_other = self.tf.reshape(obs_other, [-1, self.args.state_other_dim])
            # processed_obses = self.preprocessor.tf_process_obses(obs)
            # print('obs_ego:', obs_ego)
            # print('obs_other', obs_other)
            actions, _ = self.policy_with_value.compute_action(obs_ego, obs_other)
            # print(actions)
            obses, rewards, punish_terms_for_training, real_punish_term, veh2veh4real, veh2road4real = self.model.rollout_out(
                actions)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            punish_terms_for_training_sum += punish_terms_for_training
            real_punish_terms_sum += real_punish_term
            veh2veh4real_sum += veh2veh4real
            veh2road4real_sum += veh2road4real

        # pg loss
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_term_for_training = self.tf.reduce_mean(punish_terms_for_training_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term_for_training
        pg_loss = obj_loss + punish_loss

        real_punish_term = self.tf.reduce_mean(real_punish_terms_sum)
        veh2veh4real = self.tf.reduce_mean(veh2veh4real_sum)
        veh2road4real = self.tf.reduce_mean(veh2road4real_sum)

        return obj_loss, punish_term_for_training, punish_loss, pg_loss, \
               real_punish_term, veh2veh4real, veh2road4real, pf

    @tf.function
    def forward_and_backward(self, obs, ite, ref_index, veh_num, veh_mode):
        obj_loss, punish_term_for_training, punish_loss, pg_loss, \
        real_punish_term, veh2veh4real, veh2road4real, pf \
            = self.model_rollout_for_update(obs, ite, ref_index, veh_num, veh_mode)

        # with self.tf.name_scope('policy_gradient') as scope:
        #     pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)

        return obj_loss, \
               punish_term_for_training, punish_loss, pg_loss, \
               real_punish_term, veh2veh4real, veh2road4real, pf

    # @tf.function
    # def forward_and_backward(self, mb_obs, ite, mb_ref_index):
    #     with self.tf.GradientTape(persistent=True) as tape:
    #         obj_loss, punish_term_for_training, punish_loss, pg_loss, \
    #         real_punish_term, veh2veh4real, veh2road4real, pf\
    #             = self.model_rollout_for_update(mb_obs, ite, mb_ref_index)
    #
    #     with self.tf.name_scope('policy_gradient') as scope:
    #         pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
    #
    #     return pg_grad, obj_loss, \
    #            punish_term_for_training, punish_loss, pg_loss,\
    #            real_punish_term, veh2veh4real, veh2road4real, pf
    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    # @tf.function
    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        # mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        # mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        rewards_sum, punish_terms_for_training_sum, real_punish_terms_sum, veh2veh4real_sum, veh2road4real_sum = [], [], [], [], []
        pg_loss_total = self.tf.constant([])

        with self.tf.GradientTape(persistent=True) as tape:
            for i in range(len(self.batch_data['batch_obs'])):
                obs = self.tf.expand_dims(self.tf.constant(self.batch_data['batch_obs'][i]), axis=0)
                ref_index = self.tf.expand_dims(self.tf.constant(self.batch_data['batch_ref_index'][i], self.tf.int32),
                                                axis=0)
                veh_num = self.tf.constant(self.batch_data['batch_veh_num'][i], self.tf.int32)
                veh_mode = self.tf.constant(self.batch_data['batch_veh_mode'][i])

                with self.grad_timer:
                    obj_loss, \
                    punish_term_for_training, punish_loss, pg_loss, \
                    real_punish_term, veh2veh4real, veh2road4real, pf = \
                        self.forward_and_backward(obs, iteration, ref_index, veh_num, veh_mode)

                    rewards_sum.append(obj_loss)
                    punish_terms_for_training_sum.append(punish_term_for_training)
                    real_punish_terms_sum.append(real_punish_term)
                    veh2veh4real_sum.append(veh2veh4real)
                    veh2road4real_sum.append(veh2road4real)
                    pg_loss = self.tf.expand_dims(pg_loss, axis=0)
                    pg_loss_total = self.tf.concat([pg_loss_total, pg_loss], 0)
        pg_loss_sum = self.tf.reduce_mean(pg_loss_total, axis=0)
        print('pg_loss', pg_loss_sum)
        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(pg_loss_sum, self.policy_with_value.policy.trainable_weights)

        with self.tf.name_scope('PI_policy_gradient') as scope:
            PI_grad = tape.gradient(pg_loss_sum, self.policy_with_value.PI_policy.trainable_weights)

        pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
        PI_grad, PI_grad_norm = self.tf.clip_by_global_norm(PI_grad, self.args.gradient_clip_norm)

        # pg_grad, obj_loss, \
        #     #     punish_term_for_training, punish_loss, pg_loss, \
        # #     real_punish_term, veh2veh4real, veh2road4real, pf =\
        # #         self.forward_and_backward(mb_obs, iteration, mb_ref_index)

        # with self.grad_timer:
        #     pg_grad, obj_loss, \
        #     punish_term_for_training, punish_loss, pg_loss, \
        #     real_punish_term, veh2veh4real, veh2road4real, pf =\
        #         self.forward_and_backward(mb_obs, iteration, mb_ref_index)
        #
        #     pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term_for_training=punish_term_for_training.numpy(),
            real_punish_term=real_punish_term.numpy(),
            veh2veh4real=veh2veh4real.numpy(),
            veh2road4real=veh2road4real.numpy(),
            punish_loss=punish_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            PI_grad_norm=PI_grad_norm.numpy(),
        ))

        grads = PI_grad + pg_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
