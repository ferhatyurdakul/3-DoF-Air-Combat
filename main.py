from AirCombatEnv_2 import F16Environment
from stable_baselines3 import SAC

env = F16Environment()

curriculum_model = "sac_constant_point_following_v500k_v2"
model_name = "sac_circular_motion_following_v500k"
log_file = model_name + "_tensorboard"

train = True

model = SAC.load(curriculum_model)

if train:
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_file)

    model.learn(total_timesteps=500_000, log_interval=4)

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
