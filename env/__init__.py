from .custom_env import CustomEngine
from gym.envs.registration import register

config = {
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
    'reward_distance': 0,   # sparse reward
    'custom_observation': {'observe_robot_vel': True,
                            'observe_robot_pos': True,
                            'observe_robot_yaw': True,
                            'observe_v_pref': False,
                            'observe_robot_radius': True,
                            'observe_pillar_pos': True,
                            'observe_pillar_radius': True,
                            }
}

register(id='point-pillar-v0',
        entry_point='env.custom_env:CustomEngine',
        kwargs={'config':config})