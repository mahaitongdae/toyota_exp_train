import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

class Air3d(gym.Env):
    def __init__(self, **kwargs):
        metadata = {'render.modes': ['human']}
        self.step_length = 0.1  # ms
        self.action_number = 1
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.action_number,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.full([3,], -float('inf')),np.full([3,], float('inf')), dtype=np.float32)
        self.obs = self._reset_init_state()
        self.A = np.array([[1., 1.],[0, 1.]])
        self.B = np.array([[0],[1.]])
        self.sis_paras = None
        self.sis_info = dict()


    def reset(self):
        self.obs = self._reset_init_state()
        self.action = np.array([0.0])
        self.cstr = 0.0
        self.phi = self.adaptive_safety_index()
        self.old_d = None
        return self.obs

    def step(self, action):
        if len(action.shape) == 2:
            action = action.reshape([-1,])
        self.action = self._action_transform(action)
        reward = self.compute_reward(self.obs, self.action)
        dx = np.array([- 5. + 5. * np.cos(self.obs[2]) + action[0] * self.obs[1],
                             5. * np.sin(self.obs[2]) - action[0] * self.obs[0],
                             1. - action[0]
                             ])
        self.obs = self.obs + 0.1 * dx
        self.obs[2] = self.obs[2] % (2 * np.pi)
        constraint = 5 - np.linalg.norm(self.obs[:2])
        done = True if constraint > 0 else False
        info = dict(reward_info=dict(reward=reward, constraints=float(constraint)))
        self.cstr = constraint
        old_phi = self.phi
        self.phi = self.adaptive_safety_index()
        if old_phi <= 0:
            delta_phi = max(self.phi, 0)
        else:
            delta_phi = self.phi - old_phi
        info.update({'delta_phi':delta_phi})
        info.update(self.sis_info)
        return self.obs, reward, done, info # s' r

    def adaptive_safety_index(self, sigma=0.04, k=2, n=2):
        '''
        synthesis the safety index that ensures the valid solution
        '''
        # initialize safety index

        '''
        function phi(index::CollisionIndex, x, obs)
            o = [obs.center; [0,0]]
            d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
            dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
            dim = 2
            dp = dM[[1,dim]]
            dv = dM[[dim+1,dim*2]]
            dot_d = dp'dv / d
            return (index.margin + obs.radius)^index.phi_power - d^index.phi_power - index.dot_phi_coe*dot_d
        end
        '''
        x = self.obs
        if self.sis_paras is not None:
            sigma, k, n = self.sis_paras
        phi = -1e8
        sis_info_t = self.sis_info.get('sis_data', [])
        sis_info_tp1 = []

        rela_pos = x[:2]
        d = np.linalg.norm(rela_pos, ord=2)
        robot_to_hazard_angle = np.arctan((rela_pos[1])/(rela_pos[0] + 1e-8))\
                                + np.where(x[0] <= 0, np.pi, 0.)
        vel_rela_angle = x[-1] - robot_to_hazard_angle
        dotd = 5. * np.cos(vel_rela_angle) - 5. * np.cos(robot_to_hazard_angle)

        # if dotd <0, then we are getting closer to hazard
        sis_info_tp1.append((d, dotd))

        # compute the safety index
        phi_tmp = (sigma + 5.) ** n - d ** n - k * dotd
        # select the largest safety index
        if phi_tmp > phi:
            phi = phi_tmp

        # sis_info is a list consisting of tuples, len is num of obstacles
        self.sis_info.update(dict(sis_data=sis_info_tp1, sis_trans=(sis_info_t, sis_info_tp1)))
        return phi

    def _action_transform(self, action):
        action = 1. * np.clip(action, -1.05, 1.05)
        return action

    def compute_reward(self, obs, action):
        r = action[0] ** 2
        return r

    def _reset_init_state(self):
        x = -26. * np.random.random() + 20
        y = -20. * np.random.random() + 10
        phi = 2 * np.pi * np.random.random()
        return np.array([x, y, phi])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        extension = 1
        if mode == 'human':
            plt.ion()
            plt.cla()
            plt.title("Air3d")
            ax = plt.axes(xlim=(-6 - extension, 20 + extension),
                          ylim=(-10 - extension, 10 + extension))
            plt.axis("equal")
            plt.axis('off')

            ax.add_patch(plt.Rectangle((-6, -10), 26, 20, edgecolor='black', facecolor='none'))
            ax.add_patch(plt.Circle((0, 0), 5, edgecolor='red', facecolor='none'))
            plt.scatter(self.obs[0], self.obs[1])
            plt.arrow(self.obs[0], self.obs[1], 3 * np.cos(self.obs[2]), 3 * np.sin(self.obs[2]))

            text_x, text_y = -6, 10

            d = self.sis_info.get('sis_data')[0][0]
            dotd = self.sis_info.get('sis_data')[0][1]
            try:
                dif_d = d - self.old_d
            except:
                dif_d = d
            plt.text(text_x, text_y, 'x: {:.2f}'.format(self.obs[0]))
            plt.text(text_x, text_y - 1, 'y: {:.2f}'.format(self.obs[1]))
            plt.text(text_x, text_y - 2, 'angle: {:.2f}'.format(self.obs[2]))
            plt.text(text_x, text_y - 3, 'action: {:.2f}'.format(self.action[0]))
            plt.text(text_x, text_y - 4, 'constraints: {:.2f}'.format(self.cstr))
            plt.text(text_x, text_y - 5, 'd: {:.2f}'.format(d))
            plt.text(text_x, text_y - 6, 'dotd: {:.2f}'.format(dotd))
            plt.text(text_x, text_y - 7, 'dif_dotd: {:.2f}'.format(10 * dif_d))
            self.old_d = d

            plt.show()
            plt.pause(0.001)

def env():
    import time
    env = Air3d()
    # env.step_length = 0.01
    obs = env.reset()
    action = np.array([0.0])
    while True:
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
        if done: env.reset()

if __name__ == '__main__':
    env()










