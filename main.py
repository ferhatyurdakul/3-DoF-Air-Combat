from AirCombatEnv import F16Environment
from stable_baselines3 import SAC

env = F16Environment()

train = True

if train:
    model = SAC("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1_000_000, log_interval=4)

    model.save("sac_constant_point_following_v1")


else:
    model = SAC.load("sac_constant_point_following_v1")

    obs, info = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
