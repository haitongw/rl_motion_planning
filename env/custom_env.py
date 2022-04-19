from safety_gym.envs.engine import Engine
# from gym.utils.env_checker import check_env
from gym.spaces import Box, Discrete
import numpy as np
import pygame

def mat2yaw(mat):
    c_theta, s_theta = mat[0,0], -mat[0,1]
    yaw = np.arctan2(s_theta, c_theta)
    return yaw

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
        self.robot_radius = 0.18

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
        if self.pillars_num > 0:
            feature_num = 1 # velocity
            # if self.observe_pillar_pos:
            #     self.obs_space_dict['pillar_pos'] = Box(-np.inf, np.inf, (self.pillars_num,2), dtype=np.float32)
            # if self.observe_pillar_radius:
            #     self.obs_space_dict['pillar_radius'] = Box(0.0, 2.0, (1,), dtype=np.float32)
            if self.observe_pillar_compass:
                feature_num += 2
                # self.obs_space_dict['pillar_compass'] = Box(-np.inf, np.inf, (self.pillars_num,2), dtype=np.float32)
            if self.observe_pillar_dist:
                feature_num += 1
                # self.obs_space_dict['pillar_dist'] = Box(0.0, 1.0, (self.pillars_num,), np.float32)
            self.obs_space_dict['y_pillar_state'] = Box(-np.inf, np.inf, (self.pillars_num, feature_num), np.float32)
        if self.gremlins_num > 0:
            feature_num = 0
            if self.observe_gremlin_compass:
                feature_num += 2
            if self.observe_gremlin_dist:
                feature_num += 1
            if self.observe_gremlin_vel:
                feature_num += 1
            self.obs_space_dict['z_gremlin_state'] = Box(-np.inf, np.inf, (self.gremlins_num, feature_num), np.float32)
        if self.padding_obs:
            self.obs_space_dict['robot_vel'] = Box(-np.inf, np.inf, (1,), dtype=np.float32)
            # self.obs_space_dict['robot_radius'] = Box(0.0, 1.0, (1,), np.float32)
            # self.obs_space_dict['pillar_radius'] = Box(0.0, 2.0, (1,), dtype=np.float32)
            self.obs_space_dict['pillar_compass'] = Box(-np.inf, np.inf, (3,2), dtype=np.float32)
            self.obs_space_dict['pillar_dist'] = Box(0.0, 1.0, (3,), np.float32)
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        if self.play:
            screen = pygame.display.set_mode((640,480))
            clock = pygame.time.Clock()

    def obs_compass_without_norm(self, pos):
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        # Get ego vector in world frame
        vec = pos - self.world.robot_pos()
        # Rotate into frame
        vec = np.matmul(vec, self.world.robot_mat())
        # Truncate
        vec = vec[:self.compass_shape]
        # Normalize
        return vec

    def gremlins_obj_vel(self):        
        return [self.world.body_vel(f'gremlin{i}obj').copy() for i in range(self.gremlins_num)]

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
            # obs['goal_compass'] = self.obs_compass_without_norm(self.goal_pos)
        if self.observe_robot_pos:
            obs['robot_pos'] = self.robot_pos[:2]
        if self.observe_robot_vel:
            obs['robot_vel'] = np.array([np.hypot(*(self.world.robot_vel()[:2]))])
            if obs['robot_vel'][0] < 0.01:
                obs['robot_vel'][0] = 0
        if self.observe_robot_yaw:
            obs['robot_yaw'] = np.array([mat2yaw(self.world.robot_mat())])
        if self.observe_v_pref:
            obs['v_pref'] = self.v_pref * 10    #TODO: fix this
        # if self.observe_robot_radius:
        #     obs['robot_radius'] = np.array([self.robot_radius]) # robot radius
        # if self.observe_pillar_pos:
        #     obs['pillar_pos'] = np.array(self.pillars_pos)[:,:2]
        # if self.observe_pillar_radius:
        #     obs['pillar_radius'] = np.array([self.pillars_size])
        # if self.observe_pillar_compass:
        #     # obs['pillar_compass'] = np.array(list(map(self.obs_compass_without_norm, self.pillars_pos)))
        #     obs['pillar_compass'] = np.array(list(map(self.obs_compass, self.pillars_pos)))
        # if self.observe_pillar_dist:
        #     pillar_position = np.array(self.pillars_pos)[:,:2]
        #     distance_array = np.hypot(self.robot_pos[:2][0] - pillar_position[:,0],
        #                               self.robot_pos[:2][1] - pillar_position[:,1])
        #     obs['pillar_dist'] = np.exp(-(distance_array-0.38))
        if self.pillars_num > 0:
            obs['y_pillar_state'] = np.zeros((self.pillars_num, 4))
            if self.observe_pillar_compass:
                pillar_compass = np.array(list(map(self.obs_compass, self.pillars_pos)))
                obs['y_pillar_state'][:,:2] = pillar_compass
            if self.observe_pillar_dist:
                pillar_position = np.array(self.pillars_pos)[:,:2]
                distance_array = np.hypot(self.robot_pos[:2][0] - pillar_position[:,0],
                                        self.robot_pos[:2][1] - pillar_position[:,1])
                obs['y_pillar_state'][:,2] = np.exp(-(distance_array-0.38))
            order_idx = np.argsort(obs['y_pillar_state'][:,2])
            obs['y_pillar_state'] = obs['y_pillar_state'][order_idx]
        if self.gremlins_num > 0:
            obs['z_gremlin_state'] = np.zeros((self.gremlins_num, 4))
            if self.observe_gremlin_compass:
                gremlin_compass = np.array(list(map(self.obs_compass, self.gremlins_obj_pos)))
                obs['z_gremlin_state'][:,:2] = gremlin_compass
            if self.observe_gremlin_dist:
                gremlin_pos = np.array(self.gremlins_obj_pos)[:,:2]
                gremlin_dist = np.hypot(self.robot_pos[:2][0] - gremlin_pos[:,0],
                                        self.robot_pos[:2][1] - gremlin_pos[:,1])
                # gremlin_dist = np.array([self.dist_xy(pos) for pos in gremlin_pos])
                gremlin_dist = np.exp(-(gremlin_dist-0.28))
                obs['z_gremlin_state'][:,2] = gremlin_dist
            if self.observe_gremlin_vel:
                gremlin_vel = np.array(self.gremlins_obj_vel())[:,:2]
                gremlin_vel = np.array([np.hypot(*single_gremlin_vel) for single_gremlin_vel in gremlin_vel])
                obs['z_gremlin_state'][:,3] = gremlin_vel
            order_idx = np.argsort(obs['z_gremlin_state'][:,2])
            obs['z_gremlin_state'] = obs['z_gremlin_state'][order_idx]
        if self.padding_obs:
            obs['robot_vel'] = np.array([np.hypot(*self.world.robot_vel()[:2])])
            # obs['robot_radius'] = np.array([self.robot_radius]) # robot radius
            obs['pillar_compass'] = np.zeros((3,2))
            obs['pillar_dist'] = np.zeros((3))
            # obs['pillar_radius'] = np.array([self.pillars_size])
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

    def set_mocaps(self):
        """
        control gremlins
        """
        if self.gremlins_num: # self.constrain_gremlins:
            phase = float(self.data.time)/2.0   # 2.0
            for i in range(self.gremlins_num):
                name = f'gremlin{i}'
                target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins_travel
                pos = np.r_[target, [self.gremlins_size]]
                self.data.set_mocap_pos(name + 'mocap', pos)
        

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
            action = np.squeeze(action)
            # action = int(np.squeeze(action))
            # action = self.action_dict[action]

        obs, reward, done, info = super().step(action)

        if self.observation_flatten:
            obs_flat, obs_dict = obs
        else:
            obs_dict = obs
        
        if self.pillars_num > 0:
            pillar_position = np.array(self.pillars_pos)[:,:2]
            pillar_distance_array = np.hypot(self.robot_pos[:2][0] - pillar_position[:,0],
                                      self.robot_pos[:2][1] - pillar_position[:,1])
            # min_dist = np.min(distance_array)

            # collision
            pillar_collision_dist = pillar_distance_array - (self.robot_radius + self.pillars_size)
            min_pillar_collision_dist = np.min(pillar_collision_dist)
            # collision_dist = min_dist - (self.robot_radius + self.pillars_size)
            if min_pillar_collision_dist < 0:
                reward += -self.collision_penalty
                info['collsion'] = 1
                # print("collision")
            # too close to obstacles    
            close_penalty = 0.0
            for dist in pillar_collision_dist:
                if dist < self.too_close_dist and dist > 0:
                    info['too_close'] = 1
                    if self.nonlinear_penalty:
                        close_penalty += -self.collision_penalty * np.exp(-dist)
                    else:
                        close_penalty += float(-self.too_close_penalty + self.too_close_penalty/self.too_close_dist * dist)
            reward += close_penalty

        if self.gremlins_num > 0:
            gremlin_pos = np.array(self.gremlins_obj_pos)[:,:2]
            gremlin_dist = np.hypot(self.robot_pos[:2][0] - gremlin_pos[:,0],
                                      self.robot_pos[:2][1] - gremlin_pos[:,1])
            # collision
            gremlin_collision_dist = gremlin_dist - (self.robot_radius + self.gremlins_size)
            min_gremlin_collision_dist = np.min(gremlin_collision_dist)
            if min_gremlin_collision_dist < 0:
                reward += -self.collision_penalty
                info['collsion'] = 1
            # too close
            close_penalty = 0.0
            for dist in gremlin_collision_dist:
                if dist < self.too_close_dist and dist > 0:
                    info['too_close'] = 1
                    if self.nonlinear_penalty:
                        close_penalty += -self.collision_penalty * np.exp(-dist)
                    else:
                        close_penalty += float(-self.too_close_penalty + self.too_close_penalty/self.too_close_dist * dist)
            reward += close_penalty

            # elif collision_dist < self.too_close_dist:
            #     if self.nonlinear_penalty:
            #         penalty = -self.collision_penalty * np.exp(-collision_dist)
            #     else:
            #         penalty = float(-self.too_close_penalty + self.too_close_penalty/self.too_close_dist * collision_dist)
            #     reward += penalty
            #     # print("%.4f" % float(-self.too_close_penalty + self.too_close_penalty/self.too_close_dist * collision_dist))
            #     info['too_close'] = 1
            #     # print("too close to pillar")

        if self.observation_flatten:
            obs = obs_flat
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        if self.observation_flatten:
            obs_flat, obs_dict = obs
        else:
            obs_flat = obs
        return obs_flat
    
    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}        

if __name__ == '__main__':
    config = {
        'play': True,   # control robot from keyboard, Up, Down, Left, Right
        'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
        'placements_extents': [-2., -2., 2., 2.],  # Placement limits (min X, min Y, max X, max Y)
        'num_steps':100000000,
        'task': 'goal',
        'observation_flatten': True,
        'observe_goal_comp': True,
        'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
        'pillars_num': 3,
        'gremlins_num': 0,
        'sensors_obs': [],
        'constrain_pillars': True,
        'constrain_gremlins': False,
        'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
        'gremlins_travel': 0.8,  # Radius of the circle traveled in
        'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
        'gremlins_density': 0.001,  # Density of gremlins
        'reward_distance': 1.0,   # dense reward
        'reward_goal': 30.0,       # sparse reward
        # 'goal_placements': [(-1.5, -1.5, 1.5, 1.5)],
        'gremlins_placements': [(-2.5,-2.5,2.5,2.5)],
        'custom_observation': { 'padding_obs':False,
                                'observe_robot_vel': True,
                                'observe_robot_pos': False,
                                'observe_robot_yaw': False,
                                'observe_v_pref': False,
                                'observe_robot_radius': False,
                                'observe_pillar_pos': False,
                                'observe_pillar_radius': False,
                                'observe_pillar_compass': True,
                                'observe_pillar_dist': True,
                                'observe_gremlin_vel': False,
                                'observe_gremlin_compass': False,
                                'observe_gremlin_dist': False,
                                'collision_penalty': 0.1,
                                'too_close_penalty': 0.1,
                                'too_close_dist': 0.4,
                                'nonlinear_penalty': True,
                                }
    }

    myenv = CustomEngine(config)
    myenv.seed(42)
    act = 0
    cnt = 0
    myenv.reset()
    print(myenv.observation_space.shape)
    done = False
    for i in range(12):
        total_reward = 0
        while True:
            myenv.render()
            state, reward, done, info = myenv.step(myenv.action_space.sample())
            total_reward += reward
            if done:
                break
            cnt+=1
            if cnt >= 100:
                cnt = 0
                # print(state)
                # print("%.3f" % total_reward)
                # print("%.4f" % reward)
                # print('----------------------')

        act+=1
        obs = myenv.reset()

    myenv.close()