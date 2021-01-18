#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
import numpy as np
from gym import spaces
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        policy_model_cls = NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        # add PI network
        PI_in_dim, PI_out_dim = self.args.PI_in_dim, self.args.PI_out_dim
        n_hiddens, n_units, hidden_activation = self.args.PI_num_hidden_layers, self.args.PI_num_hidden_units, \
                                                self.args.PI_hidden_activation

        self.PI_policy = policy_model_cls(PI_in_dim, n_hiddens, n_units, hidden_activation, PI_out_dim, name='PI_policy',
                                       output_activation=self.args.PI_policy_out_activation)
        PI_policy_lr_schedule = PolynomialDecay(*self.args.PI_policy_lr_schedule)
        self.PI_policy_optimizer = self.tf.keras.optimizers.Adam(PI_policy_lr_schedule, name='adam_opt_PI')

        self.models = (self.policy,self.PI_policy)
        self.optimizers = (self.policy_optimizer, self.PI_policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        PI_policy_len = len(self.PI_policy.trainable_weights)

        PI_policy_grad, policy_grad = grads[:PI_policy_len], grads[PI_policy_len:]
        self.PI_policy_optimizer.apply_gradients(zip(PI_policy_grad, self.PI_policy.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    @tf.function
    def compute_mode(self, obs_ego, obs_other):
        obs_ego = tf.cast(obs_ego, dtype=tf.float32)
        state_other = tf.reduce_sum(self.PI_policy(obs_other), axis=0)
        state_other = tf.expand_dims(state_other, axis=0)
        state = tf.concat([obs_ego, state_other], axis=1)
        logits = self.policy(state)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    # @tf.function
    # def compute_action(self, obs):
    #     with self.tf.name_scope('compute_action') as scope:
    #         logits = self.policy(obs)
    #         if self.args.deterministic_policy:
    #             mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
    #             return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.

    @tf.function
    def compute_action(self, obs_ego, obs_other):
        with self.tf.name_scope('compute_action') as scope:
            obs_ego = tf.cast(obs_ego, dtype=tf.float32)
            state_other = tf.reduce_sum(self.PI_policy(obs_other), axis=0)
            state_other = tf.expand_dims(state_other, axis=0)
            # print(state_other)
            # print(obs_ego)
            state = tf.concat([obs_ego, state_other], axis=1)
            # print(state)
            logits = self.policy(state)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.

def test_policy():
    import gym
    from train_script import built_mixedpg_parser
    args = built_mixedpg_parser()
    print(args.obs_dim, args.act_dim)
    env = gym.make('PathTracking-v0')
    policy = PolicyWithQs(env.observation_space, env.action_space, args)
    obs = np.random.random((128, 6))
    act = np.random.random((128, 2))
    Qs = policy.compute_Qs(obs, act)
    print(Qs)

def test_policy2():
    from train_script import built_mixedpg_parser
    import gym
    args = built_mixedpg_parser()
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

def test_policy_with_Qs():
    from train_script import built_mixedpg_parser
    import gym
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
    # print(policy_with_value.policy.trainable_weights)
    # print(policy_with_value.Qs[0].trainable_weights)
    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)

    with tf.GradientTape() as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy_with_value.policy.trainable_weights)
    print(gradient)

def test_mlp():
    import tensorflow as tf
    import numpy as np
    policy = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    value = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(4,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    print(policy.trainable_variables)
    print(value.trainable_variables)
    with tf.GradientTape() as tape:
        obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
        obses = tf.convert_to_tensor(obses)
        acts = policy(obses)
        a = tf.reduce_mean(acts)
        print(acts)
        Qs = value(tf.concat([obses, acts], axis=-1))
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy.trainable_weights)
    print(gradient)


if __name__ == '__main__':
    from train_script import built_AMPC_parser, built_parser
    args = built_parser('AMPC')

