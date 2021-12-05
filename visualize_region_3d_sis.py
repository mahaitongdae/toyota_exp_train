import matplotlib.pyplot as plt

from policy import Policy4Lagrange
import os
from evaluator import Evaluator
import gym
from utils.em_brake_4test import EmergencyBraking
import numpy as np
from matplotlib.colors import ListedColormap
from dynamics.models import EmBrakeModel, UpperTriangleModel, Air3dModel, Air3dModelSis
plt.rcParams.update({'font.size': 14})

def hj_baseline():
    import jax
    import jax.numpy as jnp
    import hj_reachability as hj
    # from hj_reachability.systems import DetAir3d
    dynamics = hj.systems.DetAir3d()

    grid = hj.Grid.from_grid_definition_and_initial_values(hj.sets.Box(lo=np.array([-6., -13., 0.]),
                                                                       hi=np.array([20., 13., 2 * np.pi])),
                                                           (51, 50, 60),
                                                           periodic_dims=2)
    values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5

    solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                      hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    time = 0.
    target_time = -1.0
    target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time).block_until_ready()
    return grid, target_values

def static_region(test_dir, iteration,
                  bound=(-5., 5., -5., 5.),
                  sum=True,
                  vector=False,
                  baseline=False):
    import json
    import argparse
    import datetime
    from policy import Policy4LagrangeSis
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/sis-test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args = parser.parse_args()
    evaluator = Evaluator(Policy4LagrangeSis, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    # brake_model = EmBrakeModel()
    # double_intergrator_model = UpperTriangleModel()
    # air3d_model = Air3dModel()
    # model_dict = {"UpperTriangle": double_intergrator_model,
    #               "Air3d": air3d_model}
    # model = model_dict.get(args.env_id.split("-")[0])
    model = Air3dModelSis()


    # generate batch obses
    d = np.linspace(bound[0], bound[1], 500)
    v = np.linspace(bound[2], bound[3], 500)
    # cmaplist = ['springgreen'] * 3 + ['crimson'] * 87
    # cmap1 = ListedColormap(cmaplist)
    Dc, Vc = np.meshgrid(d, v)
    t = np.array([np.pi * 2 / 3, np.pi, np.pi * 4 / 3 ])
    # t = np.array([np.pi / 2, np.pi, np.pi * 3 / 2])
    D, V, T = np.meshgrid(d, v, t)
    flatten_d = np.reshape(D, [-1, ])
    flatten_v = np.reshape(V, [-1, ])
    flatten_t = np.reshape(T, [-1, ])
    init_obses = np.stack([flatten_d, flatten_v, flatten_t], 1)

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
    # flatten_cstr = reduced_model_rollout_for_update(init_obses)
    #
    #
    # preprocess_obs = evaluator.preprocessor.np_process_obses(init_obses)
    # flatten_mu = evaluator.policy_with_value.compute_mu(preprocess_obs).numpy()
    # flatten_cstr = np.clip(flatten_cstr, 0, np.inf)
    sis_paras = evaluator.policy_with_value.get_sis_paras.numpy()
    flatten_phi = model.adaptive_safety_index(init_obses,
                                              sigma=sis_paras[0],
                                              k=sis_paras[1],
                                              n=sis_paras[2]).numpy()

    flatten_phih = model.adaptive_safety_index(init_obses,
                                              sigma=1.0,
                                              k=2.0,
                                              n=2.0).numpy()

    flatten_phih = flatten_phih.reshape(D.shape)

    # if vector:
    #     flatten_cs = np.multiply(flatten_cstr, flatten_mu)
    # else:
    #     con_dim = -args.con_dim
    #     flatten_cstr = flatten_cstr[:, con_dim:]
    #     flatten_cs = np.multiply(flatten_cstr, flatten_mu)

    import matplotlib.pyplot as plt

    plot_items = ['phi']
    # data_dict = {'cs': flatten_cs, 'mu':flatten_mu, 'cstr': flatten_cstr}
    data_dict = {'phi':flatten_phi}
    if baseline:
        grid, target_values = hj_baseline()

    def plot_region(data_reshape, name, k, fig):
        ax = plt.subplot(1, 3, k+1)
        data_reshape = data_reshape / np.max(data_reshape)
        data_reshape += 0.15 * np.where(data_reshape==0,
                                        np.zeros_like(data_reshape),
                                        np.ones_like(data_reshape))
        ctf = ax.contourf(Dc, Vc, data_reshape, 50, cmap='rainbow')  #
        plt.axis('equal')
        plt.axis('off')
        ct1 = ax.contour(Dc, Vc, data_reshape, levels=0,
                   colors="green",
                   linewidths=3)
        ct1 = ax.contour(Dc, Vc, flatten_phih[..., k], levels=0,
                         colors="grey",
                         linewidths=3)
        # ct1.collections[0].set_label('Learned')
        x = np.linspace(0, np.pi * 2, 100)
        ax.plot(5 * np.sin(x), 5 * np.cos(x), linewidth=2, linestyle='--', label=r'$\phi_0$ safe set', color='red')
        if baseline:
            ct2 = ax.contour(grid.coordinate_vectors[0],
                       grid.coordinate_vectors[1],
                       target_values[:, :, int(10 * (k + 2))].T,
                       levels=0,
                       colors="red",
                       linewidths=3,
                       linestyle='--')
            # ct2.collections[0].set_label('HJ avoid set')
        ax.set_title(r'$\theta={:.0f}\degree$'.format(90 * (k + 1)))  # Feasibility Indicator $F(s)$,

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        name_2d = name + '_' + str(iteration) + '_2d_' + str(k) + '.jpg'
        if k == 2:
            rect0 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='red', linewidth=3, linestyle='--')
            rect1 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='grey', linewidth=3)
            rect2 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='green', linewidth=3)
            rect3 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='red', linewidth=3)

            plt.colorbar(ctf)
            # h, l = ax.get_legend_handles_labels()
            h = (rect0, rect1, rect2, rect3)
            l = (r'$\phi_0$ safe set', r'$\phi_h$ safe set', r'$\phi_{SIS}$ safe set', 'HJ avoidable set')
            fig.legend(h, l, loc='upper right')
            # plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(evaluator.log_dir, name_2d))
            # legfig, legax = plt.subplots(figsize=(7,0.75))
            # legax.set_facecolor('white')
            # leg = legax.legend(h, l, loc='center', ncol=3, handlelength=1.5,
            #            mode="expand", borderaxespad=0., prop={'size': 13})
            # legax.xaxis.set_visible(False)
            # legax.yaxis.set_visible(False)
            # for line in leg.get_lines():
            #     line.set_linewidth(4.0)
            # plt.tight_layout(pad=0.5)
            # plt.savefig(os.path.join(evaluator.log_dir, 'legend.pdf'), format='pdf')
            # plt.subplots(figsize=(2, 8))
            #
            # plt.savefig(os.path.join(evaluator.log_dir, 'color_bar.pdf'), format='pdf')




    fig = plt.figure(figsize=(12, 3))
    for plot_item in plot_items:
        data = data_dict.get(plot_item)
        if sum:
            data_k = np.sum(data, axis=1) if len(data.shape) == 2 else data
            data_reshape = data_k.reshape(D.shape)
            for k in range(data_reshape.shape[-1]):
                plot_region(data_reshape[..., k], plot_item + '_sum', k, fig)



if __name__ == '__main__':
    # static_region('./results/toyota3lane/LMAMPC-v2-2021-11-21-23-04-21', 300000)
    static_region('./results/SisAir3d/LMAMPC-sis-2021-12-05-01-17-41', 300000,
                  bound=(-6., 20., -13., 13.),
                  baseline=True) #
    # LMAMPC - vector - 2021 - 11 - 29 - 21 - 22 - 40
    # static_region('./results/uppep_triangle/LMAMPC-terminal-2021-11-30-12-40-50', 300000,
    #               bound=(-5., 5., -5., 5.),
    #               vector=False,
    #               baseline=False)  #
