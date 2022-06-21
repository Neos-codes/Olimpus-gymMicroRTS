import sys
import numpy as np
from time import perf_counter

import gym
import gym_microrts
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

np.set_printoptions(threshold=sys.maxsize)

# Crear ambiente
env = MicroRTSGridModeVecEnv(
        num_selfplay_envs = 0,
        num_bot_envs = 1,
        max_steps = 2000,
        render_theme = 1,
        ai2s = [microrts_ai.coacAI for _ in range(1)],
        map_paths = ["maps/8x8/basesWorkers8x8.xml"],
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )


obs = env.reset()


while True:
    env.render()
    action = np.zeros((64, 7))
    action_mask = env.get_action_mask()
    source_unit_mask = env.source_unit_mask.ravel()
    for i in range(len(source_unit_mask)):
        if source_unit_mask[i] == 1:
            action[i] = np.array([1, 2, 0, 0, 0, 0, 0])
            
    action = np.array(action).reshape((8, 8, -1))
    print(action)
    next_obs, reward, done, info = env.step(action)
    print("Step!")
print(action)
print(action.shape)


