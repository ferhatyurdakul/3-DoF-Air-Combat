from AirCombatEnv import F16Environment
from stable_baselines3 import SAC

env = F16Environment()

model_name = "sac_constant_point_following_v1"

train = True

if train:
    model = SAC("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=250_000, log_interval=4)

    model.save(model_name)

else:
    model = SAC.load(model_name)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
