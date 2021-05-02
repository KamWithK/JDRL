import gym
import gym_jd

env = gym.make("jd-v0", jd_path="../Game/Jelly Drift.exe")
env.step(env.action_space.sample())
env.step(env.action_space.sample())
env.step(env.action_space.sample())
env.step(env.action_space.sample())
env.step(env.action_space.sample())

while True:
    pass
