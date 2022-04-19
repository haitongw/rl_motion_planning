from .custom_env import CustomEngine
from gym.envs.registration import register

config0 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': False,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 0,   # dense reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': True,
                            'observe_robot_yaw': True,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': True,
                            'observe_pillar_radius': True,
                            'observe_pillar_compass': False,
                            }
}

register(id='point-pillar-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config0})



config1 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    # 'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': False,
    'reward_distance': 1.0,   # dense reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': False,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            }
}

register(id='point-simple-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config1})


config2 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 50.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': True,
                            'observe_pillar_compass': True,
                            }
}

register(id='point-pillar-v1',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config2})

config3 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'sensors_obs': [],
    'constrain_pillars': False,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 10.0,       # sparse reward
    'custom_observation': { 'padding_obs':True,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': True,
                            'observe_pillar_compass': False,
                            }
}

register(id='point-simple-v1',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config3})

config4 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':3000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'sensors_obs': [],
    'constrain_pillars': False,
    'reward_distance': 0.0,   # dense reward
    'reward_goal': 100.0,       # sparse reward
    'custom_observation': { 'padding_obs':True,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': True,
                            'observe_pillar_compass': False,
                            }
}

register(id='point-simple-v2',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config4})



# no robot radius, no pillar radius, with padding, with dense reward and sparse reward
config5 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'sensors_obs': [],
    'constrain_pillars': False,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 10.0,       # sparse reward
    'custom_observation': { 'padding_obs':True,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            }
}
register(id='point-simple-v3',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config5})


config6 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 10.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': True,
                            }
}
register(id='point-pillar-v2',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config6})


# no normalizatin for compass
config7 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': False,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 20.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': True,
                            }
}
register(id='point-pillar-v3',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config7})
# collision penalty: 0.1, 0.01


# normalize compass for goal and pillars
# observe normalized pillar dist
# observation shape (13,)
config8 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
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
                            'collision_penalty': 0.25,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.8,
                            }
}
register(id='point-pillar-v4',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config8})
# 0.25, 0.05, 0.2 collision cost works well

config9 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 5,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
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
                            'collision_penalty': 0.25,
                            'too_close_penalty': 0.01,
                            'too_close_dist': 0.5,
                            }
}
register(id='point-pillar-v5',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config9})


# exponential penalty for getting too close to obstacle
config10 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 1.0,    # dense reward
    'reward_goal': 30.0,       # sparse reward
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
                            'collision_penalty': 0.2,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.6,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-pillar-v6',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config10})

# exponential penalty for getting too close to obstacle
config11 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 5,
    'sensors_obs': [],
    'constrain_pillars': True,
    'reward_distance': 5.0,   # dense reward
    'reward_goal': 20.0,       # sparse reward
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
                            'collision_penalty': 0.15,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.5,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-pillar-v7',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config11})


# exponential penalty for getting too close to obstacle
config12 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'gremlins_num': 3,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            'observe_pillar_dist': False,
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.2,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.6,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config12})


# exponential penalty for getting too close to obstacle
config13 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'gremlins_num': 5,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
    # 'goal_placements': [(-1.5, -1.5, 1.5, 1.5)],
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            'observe_pillar_dist': False,
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.1,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.4,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-v1',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config13})




# exponential penalty for getting too close to obstacle
config14 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    # 'placements_extents': [-1.5, -1.5, 1.5, 1.5],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,   # 4000
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'gremlins_num': 8,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    'gremlins_placements': [(-2.5,-2.5,2.5,2.5)],
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            'observe_pillar_dist': False,
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.1,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.4,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-v2',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config14})

# exponential penalty for getting too close to obstacle
config15 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    # 'placements_extents': [-1.5, -1.5, 1.5, 1.5],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 3,
    'gremlins_num': 6,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    'gremlins_placements': [(-2.5,-2.5,2.5,2.5)],
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
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
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.1,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.4,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-pillar-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config15})


config16 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'placements_extents': [-2., -2., 2., 2.],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,
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
    'pillars_placements' : [(-2.5,-2.5,2.5,2.5)],
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
register(id='point-pillar-gru-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config16})

config17 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    'placements_extents': [-2., -2., 2., 2.],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 8,
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
    'pillars_placements' : [(-2.5,-2.5,2.5,2.5)],
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
register(id='point-pillar-gru-v1',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config17})


# exponential penalty for getting too close to obstacle
config18 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    # 'placements_extents': [-1.5, -1.5, 1.5, 1.5],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,   # 4000
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'gremlins_num': 4,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    # 'gremlins_placements': [(-1.5,-1.5,1.5,1.5)],
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            'observe_pillar_dist': False,
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.1,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.4,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-v3',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config18})


# exponential penalty for getting too close to obstacle
config19 = {
    'play': False,   # control robot from keyboard, Up, Down, Left, Right
    'robot_base': 'xmls/new_point.xml',  # Which robot XML to use as the base
    # 'placements_extents': [-1.5, -1.5, 1.5, 1.5],  # Placement limits (min X, min Y, max X, max Y)
    'num_steps':4000,   # 4000
    'task': 'goal',
    'observation_flatten': True,
    'observe_goal_comp': True,
    'observe_goal_dist': True,  # 0->1 distance closer to the goal, value closer to 1
    'pillars_num': 0,
    'gremlins_num': 12,
    'sensors_obs': [],
    'constrain_pillars': False,
    'constrain_gremlins': True,
    'gremlins_keepout': 0.5,  # Radius for keeping out (contains gremlin path)
    'gremlins_travel': 0.8,  # Radius of the circle traveled in
    'gremlins_size': 0.1,  # Half-size (radius) of gremlin objects
    'gremlins_density': 0.001,  # Density of gremlins
    'gremlins_placements': [(-3.,-3.,3.,3.)],
    'reward_distance': 1.0,   # dense reward
    'reward_goal': 30.0,       # sparse reward
    'custom_observation': { 'padding_obs':False,
                            'observe_robot_vel': True,
                            'observe_robot_pos': False,
                            'observe_robot_yaw': False,
                            'observe_v_pref': False,
                            'observe_robot_radius': False,
                            'observe_pillar_pos': False,
                            'observe_pillar_radius': False,
                            'observe_pillar_compass': False,
                            'observe_pillar_dist': False,
                            'observe_gremlin_vel': True,
                            'observe_gremlin_compass': True,
                            'observe_gremlin_dist': True,
                            'collision_penalty': 0.1,
                            'too_close_penalty': 0.1,
                            'too_close_dist': 0.4,
                            'nonlinear_penalty': True,
                            }
}
register(id='point-gremlin-v4',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config19})
