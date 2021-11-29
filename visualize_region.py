from policy import Policy4Lagrange
import os
from evaluator import Evaluator
import gym
from utils.em_brake_4test import EmergencyBraking
import numpy as np
from matplotlib.colors import ListedColormap
from dynamics.models import EmBrakeModel, UpperTriangleModel


def static_region(test_dir, iteration,
                  bound=(-5., 5., -5., 5.),
                  sum=True):
    import json
    import argparse
    import datetime
    from policy import Policy4Lagrange
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args = parser.parse_args()
    evaluator = Evaluator(Policy4Lagrange, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    brake_model = EmBrakeModel()
    double_intergrator_model = UpperTriangleModel()
    model_dict = {"UpperTriangle": double_intergrator_model}
    model = model_dict.get(args.env_id.split("-")[0])


    # generate batch obses
    d = np.linspace(bound[0], bound[1], 100)
    v = np.linspace(bound[2], bound[3], 100)
    # cmaplist = ['springgreen'] * 3 + ['crimson'] * 87
    # cmap1 = ListedColormap(cmaplist)
    D, V = np.meshgrid(d, v)
    flatten_d = np.reshape(D, [-1, ])
    flatten_v = np.reshape(V, [-1, ])
    init_obses = np.stack([flatten_d, flatten_v], 1)

    # define rollout
    def reduced_model_rollout_for_update(obses):
        model.reset(obses)
        constraints_list = []
        for step in range(args.num_rollout_list_for_policy_update[0]):
            processed_obses = evaluator.preprocessor.tf_process_obses(obses)
            actions, _ = evaluator.policy_with_value.compute_action(processed_obses)
            obses, rewards, constraints = model.rollout_out(actions)
            constraints = evaluator.tf.expand_dims(constraints, 1) if len(constraints.shape) == 1 else constraints
            constraints_list.append(constraints)
        flattern_cstr = evaluator.tf.concat(constraints_list, 1).numpy()
        return flattern_cstr
    flatten_cstr = reduced_model_rollout_for_update(init_obses)

    preprocess_obs = evaluator.preprocessor.np_process_obses(init_obses)
    flatten_mu = evaluator.policy_with_value.compute_mu(preprocess_obs).numpy()
    # flatten_cstr = np.clip(flatten_cstr, 0, np.inf)

    try:
        flatten_cs = np.multiply(flatten_cstr, flatten_mu)
    except:
        con_dim = -args.con_dim
        flatten_cstr = flatten_cstr[:, con_dim:]
        flatten_cs = np.multiply(flatten_cstr, flatten_mu)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plot_items = ['cs']
    data_dict = {'cs': flatten_cs, 'mu':flatten_mu, 'cstr': flatten_cstr}

    def plot_region(data_reshape, name):
        fig, ax = plt.subplots()
        ct = ax.contourf(D, V, data_reshape, 50, cmap='rainbow')
        plt.colorbar(ct)
        ax.contour(D, V, data_reshape, levels=0,
            colors="black",
            linewidths=3)
        name_2d = name + '_' + str(iteration) + '_2d.jpg'
        plt.savefig(os.path.join(evaluator.log_dir, name_2d))
        figure = plt.figure()
        ax = Axes3D(figure)
        ax.plot_surface(D, V, data_reshape, rstride=1, cstride=1, cmap='rainbow')
        name_3d = name + '_' + str(iteration) + '_3d.jpg'
        plt.savefig(os.path.join(evaluator.log_dir, name_3d))

    for plot_item in plot_items:
        data = data_dict.get(plot_item)
        for k in range(data.shape[1]):
            data_k = data[:, k]
            data_reshape = data_k.reshape(D.shape)
            plot_region(data_reshape, plot_item + '_' + str(k))

        if sum:
            data_k = np.sum(data, axis=1)
            data_reshape = data_k.reshape(D.shape)
            plot_region(data_reshape, plot_item + '_sum')



if __name__ == '__main__':
    # static_region('./results/toyota3lane/LMAMPC-v2-2021-11-21-23-04-21', 300000)
    static_region('./results/uppep_triangle/LMAMPC-v2-2021-11-29-01-53-25', 300000, bound=(-5., 5., -5., 5.))