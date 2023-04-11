import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch

from buffer import ReplayBuffer
from model import BayesianBehaviorAgent

# ================================ Macro ======================================
parser = argparse.ArgumentParser()

# ----------- General properties -------------
parser.add_argument('--env', type=int, default=1, help="Envrionment ID")
parser.add_argument('--max_all_steps', type=int, default=int(4e5), help="total environment steps")

parser.add_argument('--verbose', type=float, default=0, help="Verbose")
parser.add_argument('--seed', type=str, default='0', help="random seed")
parser.add_argument('--gui', type=int, default=0, help="whether to show Pybullet GUI")

parser.add_argument('--record_test', type=int, default=0, help="if True, record all testing epsiodes")

# ----------- Network hyper-parameters ----------
parser.add_argument('--z_size', type=int, default=2, help="Size of latent variable z")
parser.add_argument('--hidden_size', type=int, default=256, help="Size of other hidden layers")

# ----------- Bayes hyper-parameters -----------
parser.add_argument('--beta_z', type=float, default=100, help="coefficient of loss function of KLD of z")
parser.add_argument('--beta_x', type=float, default=0.1, help="coefficient of loss function of predicting current observation")

# ----------- RL hyper-parameters -----------
parser.add_argument('--motor_noise_beta', type=float, default=1, help="motor noise color beta (1=pink, 0=white)")

parser.add_argument('--beta_h', type=float, default=1.2, help="beta_h of SAC (alpha)")
parser.add_argument('--train_interval', type=int, default=10, help="training interval")
parser.add_argument('--step_start', type=int, default=50000, help="steps starting training")
parser.add_argument('--gamma', type=float, default=0.9, help="discount factor")

parser.add_argument('--batch_size', type=int, default=40, help="minibatch size for experience replay")
parser.add_argument('--bsmi', type=int, default=12, help="exponent of buffer size (2 ** bsmi)")

# --------------- Ablation study -------------------
parser.add_argument('--ablation', type=str, default='none', help="'none', 'next', 'bayes', 'prediction' or 'bayesandprediction'.")

# ==================== arg parse & hyper-parameter setting ==================
args = parser.parse_args()

seed = eval(args.seed)
np.random.seed(seed)
torch.manual_seed(seed)

savepath = './data/'
details_savepath = './details/'

if os.path.exists(savepath):
    print('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

if os.path.exists(details_savepath):
    print('{} exists (possibly so do data).'.format(details_savepath))
else:
    os.makedirs(details_savepath)

# ========================= T-Maze environment initialization ============================
gamma = args.gamma

rl_config = {}
vrnn_config = {}

from env.tmaze import TMazeEnv

PyBulletClientMode = 'GUI' if args.gui else 'DIRECT'
env = TMazeEnv(mode=PyBulletClientMode, obs='vision', seed=seed)
task_name = "tmaze"

max_all_steps = args.max_all_steps
max_steps = 60  # maximum steps in one episode
action_filter = lambda a: a.reshape([-1])

# =============================== Hyperparameters ================================
verbose = args.verbose
hidden_size = args.hidden_size

h_layers = [hidden_size]

rl_config["algorithm"] = "sac"
rl_config["beta_h"] = args.beta_h
rl_config["motor_noise_beta"] = args.motor_noise_beta

batch_size = args.batch_size
seq_len = max_steps

max_num_seq = int(2 ** args.bsmi)  # buffer size

record_internal_states = True
record_episode_interval = 1

step_perf_eval = int(max_all_steps / 40)

num_test_episodes = 60 

recording_steps = 6000
step_start = int(args.step_start)
step_end = max_all_steps - 2 * recording_steps
train_interval = int(args.train_interval)

input_size = env.observation_space.shape
action_size = env.action_space.shape[0]

# ==================================== Initiliaze agent and replay buffer  =================================

if env.observation_space.shape.__len__() > 1: # for imgae observations, using uint8 to store
    obs_uint8 = True
else:
    obs_uint8 = False

buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape,
                      stored_on_gpu=torch.cuda.is_available(), batch_size=batch_size,
                      max_num_seq=max_num_seq, seq_len=seq_len, obs_uint8=obs_uint8)

if args.ablation == 'none':
    agent = BayesianBehaviorAgent(input_size,
                                  action_size,
                                  hidden_size=hidden_size,
                                  z_size=args.z_size,
                                  beta_z=args.beta_z,
                                  beta_x=args.beta_x,
                                  gamma=gamma,
                                  verbose=verbose,
                                  rl_config=rl_config,
                                  vrnn_config=vrnn_config,
                                  device='cuda' if torch.cuda.is_available() else 'cpu')
elif args.ablation == 'next':
    from model_nonext import BayesianBehaviorAgentNoNext
    agent = BayesianBehaviorAgentNoNext(input_size,
                                        action_size,
                                        hidden_size=hidden_size,
                                        z_size=args.z_size,
                                        beta_z=args.beta_z,
                                        beta_x=args.beta_x,
                                        gamma=gamma,
                                        verbose=verbose,
                                        rl_config=rl_config,
                                        vrnn_config=vrnn_config,
                                        device='cuda' if torch.cuda.is_available() else 'cpu')
elif args.ablation == 'prediction':
    from model_nonpredictive import BayesianBehaviorAgentNonpredictive
    agent = BayesianBehaviorAgentNonpredictive(input_size,
                                               action_size,
                                               hidden_size=hidden_size,
                                               z_size=args.z_size,
                                               beta_z=args.beta_z,
                                               gamma=gamma,
                                               verbose=verbose,
                                               rl_config=rl_config,
                                               vrnn_config=vrnn_config,
                                               device='cuda' if torch.cuda.is_available() else 'cpu')
elif args.ablation == 'bayes':
    from model_deterministic import DeterministicAgent
    agent = DeterministicAgent(input_size,
                               action_size,
                               hidden_size=hidden_size,
                               z_size=args.z_size,
                               beta_x=args.beta_x,
                               gamma=gamma,
                               verbose=verbose,
                               rl_config=rl_config,
                               vrnn_config=vrnn_config,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
elif args.ablation == 'bayesandprediction':
    from model_deterministic import DeterministicAgent
    agent = DeterministicAgent(input_size,
                               action_size,
                               hidden_size=hidden_size,
                               z_size=args.z_size,
                               beta_x=0,
                               gamma=gamma,
                               verbose=verbose,
                               rl_config=rl_config,
                               vrnn_config=vrnn_config,
                               device='cuda' if torch.cuda.is_available() else 'cpu')

agent.record_internal_states = record_internal_states

# ====================================== Init recording data =======================
performance_wrt_step = []
global_steps = []
steps_taken_wrt_step = []

global_step = 0

SP_real_last = []
SP_real_last_stochastic_action = []	

need_to_record_performance = False

logPs = []
klds = []

episode = 0
logP = 0
kld = 0

# ===================================== Experiment Start ============================================
while global_step <= max_all_steps + 1:

    sp = env.reset()
    if isinstance(sp, tuple):   # compatible with new Gym API
        sp = sp[0].astype(np.float32)
    else:
        sp = sp.astype(np.float32)

    observations = np.zeros([max_steps + 1, *env.observation_space.shape], dtype=np.float32)
    actions = np.zeros([max_steps, *env.action_space.shape], dtype=np.float32)
    rs = np.zeros([max_steps], dtype=np.float32)
    dones = np.zeros([max_steps], dtype=np.float32)
    infos = np.zeros([max_steps + 1, 2], dtype=np.float32)  # position infomation

    t = 0
    r = 0
    observations[0] = sp
    s = None

    infos[0] = env.info['ob']  # specific for this task

    agent.init_states()

    for t in range(max_steps):
        
        start_time = time.time()

        if global_step % step_perf_eval == 0 and global_step >= step_start - 1:
            need_to_record_performance = True

        if max_all_steps - 2 * recording_steps < global_step and episode % 2 == 0:	# After training, even episodes
            action_return = 'mean'	 # using the mean action of SAC policy
        elif max_all_steps - 2 * recording_steps < global_step and episode % 2 == 1:  # After training, odd episodes
            action_return = 'normal'    # using the normal distribution of SAC policy
        else:
            action_return = 'normal'

        sp, r, done, info = agent.step_with_env(env, sp, action_return=action_return, action_filter=action_filter)
        
        infos[t + 1] = info['ob']   # specific for this task

        actions[t] = agent.a_prev.detach().cpu().numpy()
        rs[t] = r
        dones[t] = done
        observations[t + 1] = sp

        global_step += 1
        if global_step == max_all_steps - recording_steps:
            # save data
            performance_wrt_step_array = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
            global_steps_array = np.reshape(global_steps, [-1]).astype(np.float64)
            steps_taken_wrt_step_array = np.reshape(steps_taken_wrt_step, [-1]).astype(np.float64)

            logPs_array = np.array(logPs).astype(np.float64)
            klds_array = np.array(klds).astype(np.float64)

            SP_real_last_np = np.stack(SP_real_last, axis=0)
            SP_real_last_stochastic_action_np = np.stack(SP_real_last_stochastic_action, axis=0)

            data = {"max_steps": max_steps,
                    "logPs": logPs_array,
                    "klds": klds_array,
                    "SP_real_last": SP_real_last_np,
                    "SP_real_last_stochastic_action": SP_real_last_stochastic_action_np,
                    "step_perf_eval": step_perf_eval,
                    "beta_z": args.beta_z,
                    "beta_x": args.beta_x,
                    "steps_taken_wrt_step": steps_taken_wrt_step_array,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "performance_wrt_step": performance_wrt_step_array,
                    "global_steps": global_steps_array}

            sio.savemat(savepath + task_name + "_wb_{}.mat".format(seed), data, long_field_names=True)
            torch.save(agent, savepath + task_name + "_wb_{}.model".format(seed))
            print("=== Traininig complete === Model saved at {} === Testing habitual behaviors and saving episodes data at {}".format(savepath, details_savepath))

        if global_step > step_start and global_step <= step_end and global_step % train_interval == 0:
            logP, kld = agent.learn(buffer, batch_size)

            if global_step - step_start < 50:
                print("model training one step takes {} s".format(time.time() - start_time))
                start_time = time.time()

        if done or t == max_steps - 1:
            break

    # --------------------  Record Data to Buffer ----------------------
    episode_length = t + 1
    dones[t] = True

    buffer.append_episode(observations, actions, rs, dones, episode_length)

    if verbose or episode % 100 == 0:
        print(task_name + " seed {} -- episode {} (global step {}) : steps {}, total reward {}, reached position {}".format(
             seed, episode, global_step, t, np.sum(rs), infos[t + 1, :2]))

    # ---------------------- Evaluate the agent ---------------------------
    if need_to_record_performance:
        global_step_test = int((global_step // step_perf_eval) * step_perf_eval)
        
        test_return = []
        test_episode_length = []
        for e_test in range(num_test_episodes):
            infos_test = np.zeros([max_steps + 1, 2], dtype=np.float32)  # global information
            sp = env.reset()
            infos_test[0] = env.info['ob']
            if isinstance(sp, tuple):   # compatible with new Gym API
                sp = sp[0].astype(np.float32)
            else:
                sp = sp.astype(np.float32)      
            
            agent.init_states()
            test_return.append(0)
            for t_test in range(max_steps):
                sp, r, done, info = agent.step_with_env(env, sp, action_return='mean' if e_test % 2 == 0 else 'normal', action_filter=action_filter) # no motor noise in even episodes;
                infos_test[t_test + 1] = info['ob']
                test_return[-1] += r    
                if done:
                    break
            test_episode_length.append(t_test + 1)
            if args.record_test:
                agent.save_episode_data(details_savepath + task_name + "_wb_{}_test_{}_{}.mat".format(seed, global_step_test, e_test),
                                        info=infos_test[:t_test + 2])
                
        performance_wrt_step.append(np.mean(test_return))
        steps_taken_wrt_step.append(np.mean(test_episode_length))
        global_steps.append(global_step_test)
        
        logPs.append(logP)
        klds.append(kld)

        need_to_record_performance = False

        print("== Testing performance ==  " + task_name + " - seed {}, global step {}, mean return {}, mean steps taken {}".format(
              seed, global_step_test, np.mean(test_return), np.mean(test_episode_length)))

    # ------------------------- testing after training ------------------------
    if max_all_steps - 2 * recording_steps < global_step <= max_all_steps - recording_steps:
        if episode % 2 == 0:
            SP_real_last.append(infos)
        else:
            SP_real_last_stochastic_action.append(infos)
    
    if max_all_steps - recording_steps < global_step <= max_all_steps:
        if record_internal_states:
            agent.save_episode_data(details_savepath + task_name + "_wb_{}_episode_{}.mat".format(seed, episode),
                                    info=infos[:episode_length + 1])

    episode += 1
