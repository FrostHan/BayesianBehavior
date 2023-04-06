import argparse
import os

import numpy as np
import scipy.io as sio
import torch

# ================================ Macro ======================================

parser = argparse.ArgumentParser()

# ----------- General properties -------------
parser.add_argument('--seed', type=str, default='0', help="random seed")
parser.add_argument('--gui', type=int, default=0, help="Pybullet GUI")

# ------------------ Planning specific hyper-parameters ---------------------
parser.add_argument('--modelpath', type=str, default='', help="path of the used model")
parser.add_argument('--pred_seq_len', type=int, default=16, help="planning horizon steps")
parser.add_argument('--goal_marker', type=str, default='full', help="brief description of the goal. valid values: full, red, blue, less_red, less_blue")

args = parser.parse_args()
# -------------------------------
savepath = './aif/'

if os.path.exists(savepath):
    print('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

goal_marker = args.goal_marker
assert(goal_marker in ['full', 'red', 'blue', 'less_red', 'less_blue'])

# ==================== arg parse & hyper-parameter setting ==================

seed = eval(args.seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ========================  Get model path ===================

if not args.modelpath:
    modelpath = "./data/tmaze_{}.model".format(args.seed)
else:
    modelpath = args.modelpath

# ========================= Environment T-Maze ============================
rl_config = {}
vrnn_config = {}

from env.tmaze import TMazeEnv

PyBulletClientMode = 'GUI' if args.gui else 'DIRECT'
env = TMazeEnv(mode=PyBulletClientMode, obs='vision', seed=seed)
task_name = "tmaze"

max_steps = 16  # maximum steps in one episode
action_filter = lambda a: a.reshape([-1])

pred_seq_len = args.pred_seq_len

# =============================== Hyperparameters ================================

er_episodes = 6

record_internal_states = True
record_episode_interval = 1
steps_warmup = 1

input_size = env.observation_space.shape
action_size = env.action_space.shape[0]

# ==================================== Loading agent model  =================================

print("Goal-directed planning using model from {}".format(modelpath))

agent = torch.load(modelpath)

agent.record_internal_states = record_internal_states

if torch.cuda.is_available():
    agent.to(device="cuda")
    agent.device = "cuda"

# ==================================== collect data using habitual behaviors -----------

print("====== collecting data from habitual behaviors ======= ")
goal_positions = np.zeros([er_episodes, 2], dtype=np.float32)

max_steps_hab = max_steps
max_steps_aif = max_steps

S_hab = np.zeros([er_episodes, max_steps_hab + 1, *env.observation_space.shape], dtype=np.float32)
A_hab = np.zeros([er_episodes, max_steps_hab, *env.action_space.shape], dtype=np.float32)
R_hab = np.zeros([er_episodes, max_steps_hab], dtype=np.float32)
V_hab = np.zeros([er_episodes, max_steps_hab], dtype=np.float32)
I_hab = np.zeros([er_episodes, max_steps_hab + 1, 2], dtype=np.float32) # position infomation

if goal_marker == "full":
    e = 0
    count = 0

    while e < er_episodes:
        S_hab[e] = 0
        A_hab[e] = 0
        R_hab[e] = 0
        V_hab[e] = 0
        I_hab[e] = 0

        if task_name.find("maze") >= 0:

            # sp = env.reset(goal_pos=count % len(env.goal_position))  # t2maze
            if goal_marker == 'red' or goal_marker == 'less_blue':
                sp = env.reset(goal_pos=0)
            elif goal_marker == 'blue' or goal_marker == 'less_red':
                sp = env.reset(goal_pos=1)
            else:
                sp = env.reset(goal_pos=int(e >= er_episodes // 2))  # assure that half of all episodes have goal at left and the other half at right (balanced pairs).
            
            goal_positions[e, 0] = env.goal_position[0]
            goal_positions[e, 1] = env.goal_position[1]

        sp = sp.astype(np.float32)

        S_hab[e, 0] = sp
        s = None
        r = 0

        I_hab[e, 0] = env.info['ob']  

        agent.init_states()

        for t in range(max_steps_hab):
            sp, r, done, info = agent.step_with_env(env, sp, action_filter=action_filter)

            A_hab[e, t] = agent.a_prev.detach().cpu().numpy()
            S_hab[e, t + 1] = sp
            R_hab[e, t] = r
            V_hab[e, t] = 1

            I_hab[e, t + 1] = env.info['ob'] 

            if done:
                print(task_name + "-habitual: -- episode {} : used steps {},  reached position {}".format(e, t, I_hab[e, t + 1][:2]))
                e += 1
                break
        
        count += 1
        if count > 100 * er_episodes:
            print("[Run End] Habitual behavior not qualified!")
            exit(0)

    print("------------- Habitual behaviors collected ---------------")


# ============================ Goal-directed planning Start ============================================

print("====== Goal-directed planning Start ======= ")

S_aif = np.zeros([er_episodes, max_steps_aif + 1, *env.observation_space.shape], dtype=np.float32)
A_aif = np.zeros([er_episodes, max_steps_aif, *env.action_space.shape], dtype=np.float32)
R_aif = np.zeros([er_episodes, max_steps_aif], dtype=np.float32)
V_aif = np.zeros([er_episodes, max_steps_aif], dtype=np.float32)
I_aif = np.zeros([er_episodes, max_steps_aif + 1, *env.info_shape], dtype=np.float32)

e = 0

episodes_data = []

while e < er_episodes:

    if goal_marker == 'full':
        N = np.random.randint(7, int(V_hab[e].sum()) + 1)
    else:
        N = int(V_hab[e].sum())
        
    goal_y = S_hab[e, N, :]
    goal_positions[e, 0] = I_hab[e, N, 0]
    goal_positions[e, 1] = I_hab[e, N, 1]

    if goal_marker == 'red' or goal_marker == 'less_blue':
        sp = env.reset(goal_pos=0)
    elif goal_marker == 'blue' or goal_marker == 'less_red':
        sp = env.reset(goal_pos=1)
    else:
        sp = env.reset()
        env.goal_position = [goal_positions[e, 0], goal_positions[e, 1]]

    S_aif[e, 0] = sp
    s = None
    r = 0

    I_aif[e, 0] = env.info['ob'] 

    agent.init_states()

    for t in range(max_steps_aif):

        if t < steps_warmup:
            sp, r, done, info = agent.step_with_env(env, sp, action_return='mean', action_filter=action_filter)
        else:
            sp, r, done, info = agent.step_with_env_planning(env, sp, goal_y, action_return='mean', goal_marker=goal_marker, action_filter=action_filter, seq_len=pred_seq_len)

        A_aif[e, t] = agent.a_prev.detach().cpu().numpy()
        S_aif[e, t + 1] = sp
        R_aif[e, t] = r
        V_aif[e, t] = 1

        I_aif[e, t + 1] = env.info['ob']

        if done or t == max_steps_aif - 1:
            episode_length = t + 1
            print(task_name + "-goal-directed : -- episode {} : used steps {}, goal pos: {}, reached pos {}".format(e, t, goal_positions[e], I_aif[e, t + 1][:2]))
            episodes_data.append(agent.save_episode_data(None, info=I_aif[e, :episode_length + 1]))
            e += 1
            break

data = {"max_steps": max_steps,
        "A_hab": A_hab,
        "R_hab": R_hab,
        "V_hab": V_hab,
        "I_hab": I_hab,
        "A_aif": A_aif,
        "R_aif": R_aif,
        "V_aif": V_aif,
        "I_aif": I_aif,
        "goal_position": goal_positions,
        "episodes_data": episodes_data}

sio.savemat(savepath + task_name + "_aif_{}_{}.mat".format(goal_marker, seed), data, long_field_names=True)
