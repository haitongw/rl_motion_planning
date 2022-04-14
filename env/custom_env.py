import safety_gym
import gym
from safety_gym.envs.engine import Engine
# from gym.utils.env_checker import check_env
from gym.spaces import Box, Discrete
import numpy as np
import pygame

def mat2yaw(mat):
    c_theta, s_theta = mat[0,0], -mat[0,1]
    yaw = np.arctan2(s_theta, c_theta)
    return yaw

config = {
    'play': True,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':2000000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': False,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 0,
    'custom_observation': {'observe_robot_vel': True,
                            'observe_robot_pos': True,
                            'observe_robot_yaw': True,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': True,
                            'observe_pillar_radius': True,
                            }
}

class CustomEngine(Engine):

    def __init__(self, config={}):
        self.play = config['play']
        config.pop('play', None)
        self.custom_obs_config = config['custom_observation']
        config.pop('custom_observation', None)

        super().__init__(config)

        for key, value in self.custom_obs_config.items():
            setattr(self, key, value)

        self.v_pref = 0.015
        self.v_pref_meter = 0.1503
        self.w_pref = 0.3
        delta_w = 2 * self.w_pref / 5
        # rewrite action space according to the paper
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

        # rewrite observation space
        if self.observe_robot_pos:
            self.obs_space_dict['robot_pos'] = Box(-np.inf, np.inf, (2,), dtype=np.float32)
        if self.observe_robot_vel:
            self.obs_space_dict['robot_vel'] = Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_robot_yaw:
            self.obs_space_dict['robot_yaw'] = Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_v_pref:
            self.obs_space_dict['v_pref'] = Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.observe_robot_radius:
            self.obs_space_dict['robot_radius'] = Box(0.0, 1.0, (1,), np.float32)
        if self.observe_pillar_pos:
            self.obs_space_dict['pillar_pos'] = Box(-np.inf, np.inf, (self.pillars_num,2), dtype=np.float32)
        if self.observe_pillar_radius:
            self.obs_space_dict['pillar_radius'] = Box(0.0, 2.0, (1,), dtype=np.float32)
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
            print(self.observation_space)
        if self.play:
            screen = pygame.display.set_mode((640,480))
            clock = pygame.time.Clock()

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_robot_pos:
            obs['robot_pos'] = self.robot_pos[:2]
        if self.observe_robot_vel:
            obs['robot_vel'] = np.array([np.hypot(*self.world.robot_vel()[:2])])
        if self.observe_robot_yaw:
            obs['robot_yaw'] = np.array([mat2yaw(self.world.robot_mat())])
        if self.observe_v_pref:
            obs['v_pref'] = self.v_pref * 10    #TODO: fix this
        if self.observe_robot_radius:
            obs['robot_radius'] = np.array([0.18]) # robot radius
        if self.observe_pillar_pos:
            obs['pillar_pos'] = np.array(self.pillars_pos)[:,:2]
        if self.observe_pillar_radius:
            obs['pillar_radius'] = np.array([0.2])

        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            return (flat_obs, obs)
        else:
            return obs

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

        obs, reward, done, info = super().step(action)

        if self.observation_flatten:
            obs_flat, obs_dict = obs
        else:
            obs_dict = obs
            
        distance_array = np.hypot(obs_dict['robot_pos'][0] - obs_dict['pillar_pos'][:,0],
                                  obs_dict['robot_pos'][1] - obs_dict['pillar_pos'][:,1])
        min_dist = np.min(distance_array)

        # collision
        collision_dist = min_dist - (obs_dict['robot_radius'] + obs_dict['pillar_radius'])
        if collision_dist < 0:
            reward += -0.25
            info['collsion'] = 1
            # print("collision")
        elif collision_dist < 0.2:
            reward += -0.1 + 0.05 * collision_dist
            info['too_close'] = 1
            # print("too close to pillar")

        if self.observation_flatten:
            obs = obs_flat
        return obs, reward, done, info


myenv = CustomEngine(config)
myenv.seed(42)
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
        cnt+=1
        if cnt >= 100:
            cnt = 0
            # print(state)
            # print('----------------------')

    act+=1
    myenv.reset()

myenv.close()