# import safety_gym
# import gym

# env = gym.make('Safexp-PointGoal1-v0')
# env.reset()
# print(env.action_space)
# print(env.action_space.high)
# print(env.action_space.low)
# print(env.observation_space.shape)
# done = False
# for i in range(10):
#     obs = env.reset()
#     while True:
#         env.render()
#         state, reward, done, cost = env.step(env.action_space.sample())
#         if done:
#             break
# env.close()

# import argparse

# parser = argparse.ArgumentParser()

# # parser.add_argument('--t', type=float, required=True)
# parser.add_argument('--log-dir', type=str, default='abc')

# args = parser.parse_args()

# # print(args.t)
# print(args.log_dir)
# import os

# # a = os.path.expanduser("~/tmp/gym/")
# # print(a)
# os.makedirs("test_rl/")

def foo():
    return 0, 1

a = [foo() for _ in range(3)]
test_1 = [1,2,3,4,5]
test_2 = [5,4,3,2,1]
a, b, c, d, e = zip(test_1, test_2)
print(a,b,c,d,e)
# b,c = zip(*a)
# print(b)
# print(c)
# for x,y,z  in b, c:
#     print(x, y, z)