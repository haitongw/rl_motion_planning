from cProfile import label
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# # 4 gremlins
# gremlin_4_log_dir  = "./return_log/point-gremlin-v3-eval.csv"
# # 8 gremlins
# gremlin_8_log_dir = "./return_log/point-gremlin-v2-eval.csv"
# # 12 gremlins
# gremlin_12_log_dir = "./return_log/point-gremlin-v4-eval.csv"

# gremlin_4 = np.genfromtxt(gremlin_4_log_dir)
# gremlin_8 = np.genfromtxt(gremlin_8_log_dir)
# gremlin_12 = np.genfromtxt(gremlin_12_log_dir)

# # print(arr_4[0])
# plt.plot(5000*np.arange(1,36), gremlin_4[:35], label='num_obstacle: 4')
# plt.plot(5000*np.arange(1,36), gremlin_8[:35], label='num_obstacle: 8')
# plt.plot(5000*np.arange(1,36), gremlin_12, label='num_obstacle: 12')
# plt.title("environment with dynamic obstacles")
# plt.xlabel("number of interactions")
# plt.ylabel("average return")
# plt.legend()
# plt.show()

# 3 pillars
pillar_3_log_dir = "./return_log/point-pillar-gru-v0-eval.csv"
# 8 pillars
pillar_8_log_dir = "./return_log/point-pillar-gru-v1-eval.csv"

pillar_3 = np.genfromtxt(pillar_3_log_dir)
pillar_8 = np.genfromtxt(pillar_8_log_dir)

plt.plot(5000*np.arange(1,65), pillar_3, label='num_obstacle: 3')
plt.plot(5000*np.arange(1,65), pillar_8[:64], label='num_obstacle: 8')
plt.title("environment with stacle obstacle")
plt.xlabel("number of interactions")
plt.ylabel("average return")
plt.legend()
plt.show()