from AirCombatEnv import F16Environment
from stable_baselines3 import SAC

# Environment
env = F16Environment()

# File names
model_name = "sac_stationary_point_following_1M_v1"
log_file  = model_name + "/graphs"
save_file = model_name + "/saves"

train = True

if train:
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_file)
    
    model.learn(total_timesteps=1_000_000, log_interval=4)

    model.save(model_name + "/sac")

else:
    model = SAC.load(model_name + "/sac")

    obs = env.reset()

    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        env.render()
        
        if done:
            env.trajectory_plot()
            obs = env.reset()
