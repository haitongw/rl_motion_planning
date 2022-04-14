from turtle import distance
import safety_gym
import gym
from safety_gym.envs.engine import Engine
from gym.utils.env_checker import check_env
from gym.spaces import Box, Discrete
from gym.utils.play import play
import numpy as np
import pygame

def mat2yaw(mat):
    c_theta, s_theta = mat[0,0], -mat[0,1]
    yaw = np.arctan2(s_theta, c_theta)
    return yaw/np.pi * 180

config = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':2000,
    'task': 'goal',
    'observation_flatten': False,
    'observe_goal_comp': False,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 0,
}

class CustomEngine(Engine):

    def __init__(self, config={}):
        self.play = config['play']
        config.pop('play', None)
        super().__init__(config)
        self.v_pref = 0.015
        self.w_pref = 0.3
        delta_w = 2 * self.w_pref / 5
        self.action_space = Discrete(12)
        self.action_dict = {0:[self.v_pref, -self.w_pref],
                            1:[self.v_pref, -self.w_pref+delta_w],
                            2:[self.v_pref, -self.w_pref+2*delta_w],
                            3:[self.v_pref, self.w_pref-2*delta_w],
                            4:[self.v_pref, self.w_pref-delta_w],
                            5:[self.v_pref, self.w_pref],
                            6:[0.5*self.v_pref, -self.w_pref],
                            7:[0.5*self.v_pref, 0],
                            8:[0.5*self.v_pref, self.w_pref],
                            9:[0, -self.w_pref],
                            10:[0, 0],
                            11:[0, self.w_pref]}
        if self.play:
            screen = pygame.display.set_mode((640,480))
            clock = pygame.time.Clock()

    def obs(self):
        # print('customize obs')
        return super().obs()

    def step(self, action=None):
        if self.play:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    break
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action = [0.01, 0]
            elif keys[pygame.K_DOWN]:
                action = [-0.01, 0]
            elif keys[pygame.K_LEFT]:
                action = [0,0.2]
            elif keys[pygame.K_RIGHT]:
                action = [0, -0.2]
            else:
                action = [0,0]
        else:
            action = self.action_dict[action]

        obs_, reward, done, info = super().step(action)

        obs = {}
        obs['robot_pos'] = self.robot_pos[:2]
        obs['mat_yaw'] = mat2yaw(self.world.robot_mat())
        obs['goal_dist'] = obs_['goal_dist']
        obs['v_ref'] = action[0]
        obs['radius'] = 0.18 # robot radius
        obs['pillar_pos'] = np.array(super().pillars_pos)[:,:2]
        obs['pillar_radius'] = 0.2

        distance_array = np.hypot(obs['robot_pos'][0] - obs['pillar_pos'][:,0],
                                  obs['robot_pos'][1] - obs['pillar_pos'][:,1])
        min_dist = np.min(distance_array)

        # collision
        collision_dist = min_dist - (obs['radius'] + obs['pillar_radius'])
        if collision_dist < 0:
            reward += -0.25
            info['collsion'] = 1
            # print("collision")
        elif collision_dist < 0.2:
            reward += -0.1 + 0.05 * collision_dist
            info['too_close'] = 1
            # print("too close to pillar")

        return obs, reward, done, info


myenv = CustomEngine(config)

act = 0
cnt = 0
myenv.reset()
done = False
for i in range(12):
    while True:
        myenv.render()
        state, reward, done, info = myenv.step(act)
        if done:
            break

    act+=1
    myenv.reset()

myenv.close()