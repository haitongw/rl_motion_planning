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
    'custom_observation': {'observe_robot_vel': True,
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
    'custom_observation': {'observe_robot_vel': False,
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
    'reward_goal': 5.0,       # sparse reward
    'custom_observation': {'observe_robot_vel': True,
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