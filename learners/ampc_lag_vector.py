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
from dynamics.models import *

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONSTRAINTS_CLIP_MINUS = -1.0


class LMAMPCLearner2(object):
    import tensorflow as tf
    tf.config.optimizer.set_experimental_options({'constant_folding': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  })
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        brake_model = EmBrakeModel()
        double_intergrator_model = UpperTriangleModel()
        air3d_model = Air3dModel()
        model_dict = {"UpperTriangle": double_intergrator_model,
                      "Air3d": air3d_model}
        self.model = model_dict.get(args.env_id.split("-")[0])
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        # self.constraint_total_dim = args.num_rollout_list_for_policy_update[0] * self.model.constraints_num

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32)
                           # 'batch_ref_index': batch_data[5].astype(np.int32)
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

    def model_rollout_for_update(self, start_obses, ite):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        # cs_sum = self.tf.zeros((start_obses.shape[0]))
        # constraints_list = self.tf.TensorArray(self.tf.float32, size=self.num_rollout_list_for_policy_update[0])
        constraints_list = []
        # mu = self.policy_with_value.compute_mu(obses)
        # mu_clip = self.tf.clip_by_value(mu, 0, self.args.mu_clip_value)
        # constraints_all = self.tf.zeros((start_obses.shape[0],self.constraint_total_dim ))
        # con_dim = self.model.constraints_num
        for step in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, rewards, constraints = self.model.rollout_out(actions)
            constraints_clip = self.tf.clip_by_value(constraints, CONSTRAINTS_CLIP_MINUS, 100)
            constraints_clip = self.tf.expand_dims(constraints_clip, axis=1) if len(constraints_clip.shape) == 1 else constraints_clip
            constraints_list.append(constraints_clip)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            # cs_sum += self.tf.reduce_sum(self.tf.multiply(mu_clip, self.tf.stop_gradient(constraints_clip)), 1)
            # punish_terms_sum += self.tf.reduce_sum(self.tf.multiply(self.tf.stop_gradient(mu_clip), constraints_clip),1)
            # constraints_sum += constraints
        # constraints_all = self.tf.transpose(constraints_list.stack())
        constraints_all =self.tf.concat(constraints_list, 1)
        processed_start_obses = self.preprocessor.tf_process_obses(start_obses)
        mu_all = self.policy_with_value.compute_mu(processed_start_obses)
        cs_sum = self.tf.reduce_sum(self.tf.multiply(mu_all, self.tf.stop_gradient(constraints_all)), 1)
        punish_terms_sum = self.tf.reduce_sum(self.tf.multiply(self.tf.stop_gradient(mu_all), constraints_all),1)
        terminal_mu = mu_all[:, -1]
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_terms = self.tf.reduce_mean(punish_terms_sum)
        pg_loss = obj_loss + punish_terms
        cs_loss = -self.tf.reduce_mean(cs_sum)
        constraints = self.tf.reduce_mean(constraints_all)

        return obj_loss, punish_terms, cs_loss, pg_loss, constraints, terminal_mu

    @tf.function
    def forward_and_backward(self, mb_obs, ite):
        with self.tf.GradientTape(persistent=True) as tape:
            obj_loss, punish_terms, cs_loss, pg_loss, constraints, terminal_mu\
                = self.model_rollout_for_update(mb_obs, ite)

        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
            mu_grad = tape.gradient(cs_loss, self.policy_with_value.mu.trainable_weights) #TODO: why use -pg_loss here lead to no grad?

        return pg_grad, mu_grad, obj_loss, punish_terms, cs_loss, pg_loss, constraints, terminal_mu

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        # mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            pg_grad, mu_grad, obj_loss, punish_terms, cs_loss, pg_loss, constraints, terminal_mu =\
                self.forward_and_backward(mb_obs, iteration)

            obj_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
            mu_grad, mu_grad_norm = self.tf.clip_by_global_norm(mu_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_terms=punish_terms.numpy(),
            # real_punish_term=real_punish_term.numpy(),
            # veh2veh4real=veh2veh4real.numpy(),
            # veh2road4real=veh2road4real.numpy(),
            constraints=constraints.numpy(),
            cs_loss=cs_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            mu_grad_norm=mu_grad_norm.numpy(),
            mean_terminal_mu=np.mean(terminal_mu.numpy()),
            max_terminal_mu=np.max(terminal_mu.numpy()),
            min_terminal_mu=np.min(terminal_mu.numpy()),
        ))

        grads = obj_grad  + mu_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass