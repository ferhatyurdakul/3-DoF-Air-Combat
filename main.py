from AirCombatEnv import F16Environment as Stage1Env
from AirCombatEnv_2 import F16Environment as Stage2Env
from AirCombatEnv_3 import F16Environment as Stage3Env
from AirCombatEnv_4 import F16Environment as SubStageEnv
from stable_baselines3 import SAC

# ---------- PARAMETERS ----------
# Environment (Choices -> Stage1Env, Stage2Env, Stage3Env, SubStageEnv)
env = SubStageEnv()

# Model name to be saved or evaluated
model_name = "models/sac_dofight_1M_v2"

# Train (True) / Evaluation (False)
train = False

# To render or not during the evaluation
render = True 

# To do curriculum or not
# (Stage1 -> sac_stationary_point_following_1M_v3) 
# (Stage2 -> sac_moving_target_following_1M_v3) 
# (Stage3 -> sac_dofight_1M_v1) 
# (Substage -> sac_dofight_1M_v2) 
curriculum = False
curriculum_model = "models/sac_stationary_point_following_1M_v3" 


# ---------- CONSTANTS ----------
log_file  = model_name + "/graphs"
save_file = model_name + "/saves"


# ---------- TRAIN/EVALUATION ----------
if curriculum:
    model = SAC.load(curriculum_model + "/sac")

if train:
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_file)
    
    model.learn(total_timesteps=1_000_000, log_interval=4)

    model.save(model_name + "/sac")

else:
    model = SAC.load(model_name + "/sac")

    obs = env.reset()

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()
        
        if done:
            env.trajectory_plot()
            obs = env.reset()
