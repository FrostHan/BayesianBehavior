import warnings
import time
import torch
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch import distributions as dis
import scipy.io as sio
from torch.nn import functional as F
from base_modules import *
import colorednoise  # Use Pink Noise

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 10
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor

torch.set_default_dtype(torch.float32)

def softplus(x, min_val=1e-2):
    return torch.log(1 + torch.exp(x)) + min_val

def compute_kld(mu_q, sig_q, mu_p, sig_p, v=1, w=1, keep_batch=False):
    loss_batch = torch.mean(torch.sum(torch.log(sig_p) - torch.log(sig_q) + ((mu_p - mu_q).pow(2) + sig_q.pow(2))
                                      / (2.0 * sig_p.pow(2)) - 0.5, dim=-1) * v, dim=-1)
    if keep_batch:
        return loss_batch
    else:
        return torch.mean(loss_batch * w)

class BayesianBehaviorAgentNoNext(nn.Module):
    def __init__(self,
                 input_size,
                 action_size,
                 output_size=None,
                 hidden_size=256,
                 beta_z=100,
                 beta_x=0.1,
                 z_size=2,
                 gamma=0.9,
                 verbose=0,
                 vrnn_config=None,
                 rl_config=None,
                 is_main_network=True,
                 device='cuda'):

        super(BayesianBehaviorAgentNoNext, self).__init__()

        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = device
        
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.gamma = gamma

        self.beta_z = beta_z
        self.beta_x = beta_x

        self.verbose = verbose

        self.input_size = input_size
        self.action_size = action_size
        self.output_size = input_size if output_size is None else output_size

        self.record_internal_states = False

        # ================================== VRNN configuration ================================
        if vrnn_config is None:
            vrnn_config = {}

        self.decode_layers = vrnn_config["decode_layers"] if ("decode_layers" in vrnn_config) else [hidden_size, hidden_size]
        self.x_phi_layers = vrnn_config["x_phi_layers"] if ("x_phi_layers" in vrnn_config) else [hidden_size, hidden_size]
        self.h2z_layers = vrnn_config["h2z_layers"] if ("h2z_layers" in vrnn_config) else [hidden_size, hidden_size]

        self.sig_min_x = vrnn_config["sig_min_x"] if ("sig_min_x" in vrnn_config) else 1e-3
        self.sig_min_z = vrnn_config["sig_min_z"] if ("sig_min_z" in vrnn_config) else 1e-3

        # ================================= RL algorithm configuration =============================
        if rl_config is None:
            rl_config = {}

        # ------ common ------
        self.algorithm = rl_config["algorithm"] if "algorithm" in rl_config else "sac"

        self.policy_layers = rl_config["policy_layers"] if ("policy_layers" in rl_config) else [hidden_size, hidden_size]
        self.value_layers = rl_config["value_layers"] if ("value_layers" in rl_config) else [hidden_size, hidden_size]

        self.motor_noise_beta = rl_config["motor_noise_beta"] if ("motor_noise_beta" in rl_config) else 1  # beta of colored motor noise. 0:Gaussian noise, 1:Pink noise 

        # =================================== CNN part  ==================================

        st_cnn_v, self.input_feature_size = make_cnn(self.input_size[0])  # for value function network
        st_cnn_z, self.input_feature_size = make_cnn(self.input_size[0])  # for z encoding network
        dcnn = make_dcnn(self.decode_layers[-1], self.input_size[0])
        dcnn_next = make_dcnn(self.decode_layers[-1], self.input_size[0])

        # ==================================== VRNN part ==================================
        feedforward_actfun_rnn = nn.Tanh
        feedforward_actfun_fnn = nn.ReLU

        self.main_rnn_module = nn.ModuleList()
        self.policy_module = nn.ModuleList()  
        self.value_module = nn.ModuleList()
        
        # ------------ feature extraction mlp --------------

        self.f_x2phi_z = nn.Sequential(st_cnn_z, make_mlp(self.input_feature_size, self.x_phi_layers[:-1], self.x_phi_layers[-1], feedforward_actfun_fnn))
        self.f_x2phi_v = nn.Sequential(st_cnn_v, make_mlp(self.input_feature_size, self.x_phi_layers[:-1], self.x_phi_layers[-1], feedforward_actfun_fnn))

        self.main_rnn_module.append(self.f_x2phi_z) 
        self.value_module.append(self.f_x2phi_v)

        # ------------ RNNs ------------------
        # value network
        self.rnn_v = nn.GRU(input_size=self.x_phi_layers[-1], hidden_size=hidden_size, batch_first=True)
        self.value_module.append(self.rnn_v)

        # Main RNN
        self.rnn_h = nn.GRU(input_size=self.z_size, hidden_size=hidden_size, batch_first=True)
        self.main_rnn_module.append(self.rnn_h)

        # policy network
        self.rnn_a = nn.GRU(input_size=self.hidden_size, hidden_size=hidden_size, batch_first=True)
        self.policy_module.append(self.rnn_a)

        # ------------ output decoder ----------------
        decoder = make_mlp(self.hidden_size, self.decode_layers[:-1], self.decode_layers[-1], feedforward_actfun_fnn)
        decoder_next = make_mlp(self.hidden_size, self.decode_layers[:-1], self.decode_layers[-1], feedforward_actfun_fnn)

        self.f_h2muy = nn.Sequential(decoder, UnsqueezeModule(-1), UnsqueezeModule(-1), dcnn)
        self.f_h2muy_next = nn.Sequential(decoder_next, UnsqueezeModule(-1), UnsqueezeModule(-1), dcnn_next)

        self.main_rnn_module.append(self.f_h2muy)
        self.main_rnn_module.append(self.f_h2muy_next)

        # ------------- compute posterior z  ------------
        pre_zq_input_size = hidden_size + hidden_size  # RNN state concat with observation embedding

        self.f_h2muz_q = make_mlp(pre_zq_input_size, self.h2z_layers, z_size, feedforward_actfun_rnn, last_layer_linear=True)
        self.main_rnn_module.append(self.f_h2muz_q)

        self.f_h2aspsigz_q = nn.Sequential(make_mlp(pre_zq_input_size, self.h2z_layers, z_size, feedforward_actfun_rnn, last_layer_linear=True))
        self.main_rnn_module.append(self.f_h2aspsigz_q)

        # ===================================== RL part ===================================

        if self.algorithm == 'sac':
            self.target_entropy = rl_config["target_entropy"] if ("target_entropy" in rl_config) else np.float32(
                - self.action_size)
            self.alg_type = 'actor_critic'
            self.lr_rl = rl_config["lr_rl"] if ("lr_rl" in rl_config) else 3e-4
            self.beta_h = rl_config["beta_h"] if ("beta_h" in rl_config) else 'auto_1.0'
            self.a_coef = rl_config["a_coef"] if ("a_coef" in rl_config) else 30

            if isinstance(self.beta_h, str) and self.beta_h.startswith('auto'):
                # Default initial value of beta_h when learned
                init_value = 1.0
                if '_' in self.beta_h:
                    init_value = float(self.beta_h.split('_')[1])
                    assert init_value > 0., "The initial value of beta_h must be greater than 0"
                self.log_beta_h = torch.tensor(np.log(init_value).astype(np.float32), requires_grad=True)
            else:
                self.beta_h = float(self.beta_h)

            if isinstance(self.beta_h, str):
                self.optimizer_e = torch.optim.Adam([self.log_beta_h], lr=self.lr_rl)  # optimizer for beta_h

            # policy network
            value_input_size = self.hidden_size
            policy_input_size = self.hidden_size
            
            self.f_s2pi0 = ContinuousActionPolicyNetwork(policy_input_size, self.action_size, hidden_layers=self.policy_layers)
            self.policy_module.append(self.f_s2pi0)
            
            # V network
            self.f_s2v = ContinuousActionVNetwork(value_input_size, hidden_layers=self.value_layers)

            # Q network 1
            self.f_sa2q1 = ContinuousActionQNetwork(value_input_size, self.action_size, hidden_layers=self.value_layers) 
            # Q network 2
            self.f_sa2q2 = ContinuousActionQNetwork(value_input_size, self.action_size, hidden_layers=self.value_layers)
            
            self.value_module.append(self.f_s2v)
            self.value_module.append(self.f_sa2q1)
            self.value_module.append(self.f_sa2q2)

            self.optimizer = torch.optim.Adam([*self.value_module.parameters(), *self.policy_module.parameters(),
                                               *self.main_rnn_module.parameters()], lr=self.lr_rl)
        else:
            raise NotImplementedError

        self.mse_loss = nn.MSELoss(reduction='none')

        self.rl_update_times = 0

        # ==================================== target network synchronization ================================
        if is_main_network and self.algorithm == "sac":
            target_net = deepcopy(self)
            # synchronizing target network and main network
            state_dict_tar = target_net.state_dict()
            state_dict = self.state_dict()
            for key in list(target_net.state_dict().keys()):
                state_dict_tar[key] = state_dict[key]
            target_net.load_state_dict(state_dict_tar)
            self.target_net = target_net

        # -------------------- init variables  --------------------
        self.h = torch.zeros([1, hidden_size], dtype=torch.float32, device=self.device)  # main RNN state
        self.h_a = torch.zeros([1, hidden_size], dtype=torch.float32, device=self.device)  # policy RNN state
        self.a_prev = torch.zeros([1, self.action_size], dtype=torch.float32)  # last action

        # Pink noise motor exploration
        self.colored_noise_episode = np.zeros([10000, self.action_size], dtype=np.float32)
        for i in range(self.colored_noise_episode.shape[-1]):
            self.colored_noise_episode[:, i] = colorednoise.powerlaw_psd_gaussian(self.motor_noise_beta, 10000).astype(np.float32)
        self.env_step = 0
        
        # -------------------- variables to be recorded --------------------
        self.n_levels = 1
        self.init_recording_variables()
        if self.record_internal_states:
            self.init_recording_variables()
        self.to(device=self.device)

    @staticmethod
    def sample_z(mu, sig):
        # Using reparameterization trick to sample from a gaussian
        if isinstance(sig, torch.Tensor):
            eps = Variable(torch.randn_like(mu))
        else:
            eps = torch.randn_like(mu)
        return mu + sig * eps

    def pred_muy(self, h0_t):
        # suppose the image prediction y satisfied 0 <= y <= 1, muy is the Bournulli variable such that y=sigmoid(muy)
        if len(h0_t.size()) == 3:  # batch x sequence x hidden_size
            batch_size = h0_t.shape[0]
            sequence_length = h0_t.shape[1]
            h0_t = h0_t.reshape([batch_size * sequence_length, *h0_t.shape[2:]])
            last = h0_t
            muy = self.f_h2muy(last)
            muy = muy.reshape([batch_size, sequence_length, *muy.shape[1:]])

        else:  # batch x hidden_size
            last = h0_t
            muy = self.f_h2muy(last)

        return muy
    
    def pred_muy_next(self, h0_t):
        if len(h0_t.size()) == 3:  # batch x sequence x hidden_size
            batch_size = h0_t.shape[0]
            sequence_length = h0_t.shape[1]
            h0_t = h0_t.reshape([batch_size * sequence_length, *h0_t.shape[2:]])
            last = h0_t
            muy = self.f_h2muy_next(last)
            muy = muy.reshape([batch_size, sequence_length, *muy.shape[1:]])

        else:  # batch x hidden_size
            last = h0_t
            muy = self.f_h2muy_next(last)

        return muy
    
    def init_states(self):

        with torch.no_grad():
            self.noise = np.random.normal(0, 1, size=[self.action_size])
            self.h = torch.zeros_like(self.h)
            self.h_a = torch.zeros_like(self.h_a)

            self.z_p = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)
            self.z_q = torch.zeros([1, self.z_size], dtype=torch.float32, device=self.device)

            self.h_levels, self.c_levels = [self.h], [self.h_a]
            self.a_prev = torch.zeros([1, self.action_size], dtype=torch.float32)

            self.colored_noise_episode = np.zeros([10000, self.action_size], dtype=np.float32)
            for i in range(self.colored_noise_episode.shape[-1]):
                self.colored_noise_episode[:, i] = colorednoise.powerlaw_psd_gaussian(self.motor_noise_beta, 10000).astype(np.float32)
            self.env_step = 0
        
        if self.record_internal_states:
            self.init_recording_variables()

    def compute_prior_z(self, h_tm1, colored_noise=False):

        mu_p_t = torch.zeros([*h_tm1.shape[:-1], self.z_size], dtype=torch.float32, device=self.device)
        sig_p_t = torch.ones([*h_tm1.shape[:-1], self.z_size], dtype=torch.float32, device=self.device)        
        z_p_t = self.sample_z(mu_p_t, sig_p_t)

        return z_p_t, mu_p_t, sig_p_t

    def compute_posterior_z(self, h_tm1, x_t_obs, x_tp1_obs):
        
        if x_t_obs.shape.__len__() >= 5:
            x_reshaped = x_t_obs.reshape([-1, self.input_size[-3], self.input_size[-2], self.input_size[-1]])
            x_phi_z = self.f_x2phi_z(x_reshaped).reshape([-1, self.hidden_size])
        else:
            x_phi_z = self.f_x2phi_z(x_t_obs)

        last = torch.cat([h_tm1, x_phi_z], dim=-1)
        
        mu_q_t = self.f_h2muz_q(last)
        sig_q_t = softplus(self.f_h2aspsigz_q(last))
        
        z_q_t = self.sample_z(mu_q_t, sig_q_t)
        
        if x_t_obs.shape.__len__() >= 5:
            mu_q_t = mu_q_t.reshape([x_t_obs.shape[0], x_t_obs.shape[1], -1])
            sig_q_t = sig_q_t.reshape([x_t_obs.shape[0], x_t_obs.shape[1], -1])
            z_q_t = z_q_t.reshape([x_t_obs.shape[0], x_t_obs.shape[1], -1])

        return z_q_t, mu_q_t, sig_q_t

    def step_with_env(self, env, x_prev, action_return='normal', action_filter=None, prior_z_funtion=None):

        with torch.no_grad():
            # ------------------ Compute action ---------------------

            muy_pred = self.pred_muy_next(self.h)

            self.z_p, self.mu_z_p, self.sig_z_p = self.compute_prior_z(self.h)

            if prior_z_funtion is not None:
                self.z_p = prior_z_funtion(self.h)
                
            # main RNN forward using prior z (habitual intension)
            _, h_habitual = self.rnn_h(torch.unsqueeze(self.z_p, 1), torch.unsqueeze(self.h, 0))

            # policy RNN forward
            _, h_a = self.rnn_a(torch.unsqueeze(h_habitual[0], 1), torch.unsqueeze(self.h_a, 0))
            self.h_a = h_a[0]

            if self.algorithm == "sac":
                mua, logsiga = self.f_s2pi0(self.h_a)
                siga = torch.exp(logsiga)
            else:
                raise NotImplementedError
            
            if action_return == 'normal':
                self.noise = self.colored_noise_episode[self.env_step]
                u = mua + torch.from_numpy(self.noise).to(device=self.device) * siga
            elif action_return == 'mean':
                u = mua
            else:
                raise NotImplementedError
            
            self.env_step += 1

            if self.algorithm == "sac":
                a = torch.tanh(u)
            else:
                raise NotImplementedError
            
            self.a_prev = a

            # -------------- interact with env --------------
            a = a.detach().cpu().numpy()
            if action_filter:
                a = action_filter(a)
            
            results = env.step(a)
            if len(results) == 4:  # old Gym API
                x_curr, r_prev, done, info = results
            else:  # New Gym API
                x_curr, r_prev, terminated, truncated, info = results
                done = terminated or truncated

            # -------------- forward inference --------------
            x_prev_tensor = torch.from_numpy(x_prev)
            if len(x_prev.shape) in [1, 3]:
                x_prev_tensor = x_prev_tensor.reshape([1, *x_prev.shape]).to(torch.float32).to(self.device)
            
            x_curr_tensor = torch.from_numpy(x_curr)
            if len(x_curr.shape) in [1, 3]:
                x_curr_tensor = x_curr_tensor.reshape([1, *x_curr.shape]).to(torch.float32).to(self.device)
            
            self.z_q, self.mu_z_q, self.sig_z_q = self.compute_posterior_z(self.h, x_prev_tensor, x_curr_tensor)

            if prior_z_funtion is not None:
                self.z_q = self.z_p

            _, h_posterior = self.rnn_h(torch.unsqueeze(self.z_q, 1), torch.unsqueeze(self.h, 0))
            self.h = h_posterior[0]

        if self.record_internal_states:
            self.model_h_series.append(self.h.detach().cpu().numpy())
            self.model_h_a_series.append(self.h_a.detach().cpu().numpy())
            self.model_z_p_series.append(self.z_p.detach().cpu().numpy())
            self.model_z_q_series.append(self.z_q.detach().cpu().numpy())
            self.model_mu_z_q_series.append(self.mu_z_q.detach().cpu().numpy())
            self.model_sig_z_q_series.append(self.sig_z_q.detach().cpu().numpy())
            self.model_sig_z_p_series.append(self.sig_z_p.detach().cpu().numpy())
            self.model_mu_z_p_series.append(self.mu_z_p.detach().cpu().numpy())
        
            pred_vision = muy_pred.reshape(self.input_size)
            self.pred_visions.append(pred_vision.detach().cpu().numpy())
            if len(self.obs_series) == 0:
                self.obs_series.append(x_prev)
            self.obs_series.append(x_curr)
            self.a_series.append(a)
            self.r_series.append(r_prev)
            # self.u_series.append(u.detach().cpu().numpy())
            self.mua_series.append(mua.detach().cpu().numpy())
            self.siga_series.append(siga.detach().cpu().numpy())

        return x_curr, r_prev, done, info

    def init_recording_variables(self):
        self.model_h_series = []
        self.model_h_a_series = []
        self.model_mu_z_q_series = []
        self.model_sig_z_q_series = []
        self.model_mu_z_p_series = []
        self.model_sig_z_p_series = []
        self.model_z_q_series = []
        self.model_z_p_series = []
        self.model_mu_s_q_series = []
        self.model_sig_s_q_series = []
        self.model_s_q_series = []

        self.obs_series = []
        self.r_series = []
        self.a_series = []
        self.mua_series = []
        self.siga_series = []
        self.pred_visions = []

        # --------------- For Active Inference --------------
        self.step_weighting_series = []
        self.pred_trajectories = []
        self.z_aif_batch_series = []
        # self.h_aif_batch_series = []

    
    def save_episode_data(self, filename=None, info=None):
        data = {}

        data['model_h_a'] = np.array(self.model_h_a_series).squeeze()
        data['model_h'] = np.array(self.model_h_series).squeeze()
        data['model_z_q'] = np.array(self.model_z_q_series).squeeze()
        data['model_z_p'] = np.array(self.model_z_p_series).squeeze()
        data['model_sig_z_q'] = np.array(self.model_sig_z_q_series).squeeze()
        data['model_mu_z_q'] = np.array(self.model_mu_z_q_series).squeeze()
        data['model_sig_z_p'] = np.array(self.model_sig_z_p_series).squeeze()
        data['model_mu_z_p'] = np.array(self.model_mu_z_p_series).squeeze()

        data['obs'] = np.array(self.obs_series).squeeze()
        data['reward'] = np.array(self.r_series).squeeze()
        data['action'] = np.array(self.a_series).squeeze()
        data['mua'] = np.array(self.mua_series).squeeze()
        data['siga'] = np.array(self.siga_series).squeeze()
        data['step_weighting_series'] = np.array(self.step_weighting_series).squeeze()
        data['pred_trajectories'] = np.array(self.pred_trajectories)
        data['z_aif_batch'] = np.array(self.z_aif_batch_series)
        # data['h_aif_batch'] = np.array(self.h_aif_batch_series)

        try:
            data['pred_visions'] = np.array(self.pred_visions)
            data['goal_obs'] = np.array(self.goal_obs)
        except:
            pass

        if info is not None:
            data['info'] = info

        if filename is None:
            return data
        else:
            return sio.savemat(filename, data)

    def learn(self, buffer, minibatch_size):

        obs_batch, action_batch, reward_batch, done_batch, mask_batch, length_batch = buffer.sample_batch()

        max_stps = int(torch.max(length_batch))

        x_batch = obs_batch[:, :max_stps + 1]
        yp_batch = obs_batch[:, :max_stps + 1]

        a_batch = action_batch[:, :max_stps]
        r_batch = reward_batch[:, :max_stps]
        d_batch = done_batch[:, :max_stps].to(torch.float32)
        mask_batch = mask_batch[:, :max_stps].to(torch.float32)

        maskp_batch = torch.cat([torch.ones_like(mask_batch[:, 0:1]), mask_batch], dim=1)

        w_batch = torch.ones_like(action_batch[:, 0, :1])  # weighting of each sample in the batch, here is uniform

        # ----------------- obtain initial internal states --------------------------------------
        # main network
        h_beg = torch.zeros([minibatch_size, self.hidden_size], dtype=torch.float32, device=self.device)
        # Policy network
        ha_beg = torch.zeros([minibatch_size, self.hidden_size], dtype=torch.float32, device=self.device)

        # Value network
        hv_beg = torch.zeros([minibatch_size, self.hidden_size], dtype=torch.float32, device=self.device)

        hv_tar_beg = hv_beg.detach()

        # Burn-in (like in R2D2) for RNN not used 
        
        # ----------------------------- model training --------------------------
        KL_loss = 0
        KL_weight = 1. / self.action_size

        # Predicive RNN

        xp_tensor = torch.cat([x_batch[:, 1:], torch.zeros_like(x_batch[:, :1])], dim=1).detach()
        x_tensor = x_batch

        z_q_series = []
        muz_q_series = []
        sigz_q_series = []
        h_q_series = []

        h_t = h_beg.view([1, *h_beg.shape])
        for stp in range(max_stps + 1):
            z_q_t, muz_q_t, sigz_q_t = self.compute_posterior_z(h_t[0], x_tensor[:, stp], xp_tensor[:, stp])
            _, h_t = self.rnn_h(torch.unsqueeze(z_q_t, 1), h_t)
            
            z_q_series.append(z_q_t)
            muz_q_series.append(muz_q_t)
            sigz_q_series.append(sigz_q_t)
            h_q_series.append(h_t[0])
        
        z_q_tensor = torch.stack(z_q_series, dim=1)
        muz_q_tensor = torch.stack(muz_q_series, dim=1)
        sigz_q_tensor = torch.stack(sigz_q_series, dim=1)
        h_q_tensor = torch.stack(h_q_series, dim=1)

        h_q_tensor_tm1 = torch.cat([h_beg.reshape([h_beg.shape[0], 1, h_beg.shape[1]]), h_q_tensor[:, :-1]], dim=1)
        z_p_tensor, muz_p_tensor, sigz_p_tensor = self.compute_prior_z(h_q_tensor_tm1)
        
        muy_pred_tensor = self.pred_muy(h_q_tensor)
        muyp_pred_tensor = self.pred_muy_next(h_q_tensor_tm1)
        
        mask_kld = torch.cat([mask_batch, torch.zeros_like(mask_batch[:, 0:1])], dim=1)
        kld_z = compute_kld(muz_q_tensor, sigz_q_tensor, muz_p_tensor, sigz_p_tensor, mask_kld, w_batch)  # valid
        kld = kld_z

        KL_loss = KL_weight * self.beta_z * kld_z

        obs_dims = [-1, -2, -3] 

        KL_x_t = torch.mean((
            torch.sum(F.binary_cross_entropy_with_logits(muy_pred_tensor, yp_batch, reduction='none')
                        + yp_batch * torch.log(yp_batch.clip(1e-9, 1)) + (1 - yp_batch) * torch.log((1 - yp_batch).clip(1e-9, 1)),
            dim=obs_dims) * maskp_batch).mean(dim=-1) * w_batch)
        
        # KL_x_tp1 = torch.mean((
        #     torch.sum(F.binary_cross_entropy_with_logits(muyp_pred_tensor, yp_batch, reduction='none')
        #                 + yp_batch * torch.log(yp_batch.clip(1e-9, 1)) + (1 - yp_batch) * torch.log((1 - yp_batch).clip(1e-9, 1)),
        #     dim=obs_dims) * maskp_batch).mean(dim=-1) * w_batch)
        
        KL_x = KL_x_t 
        loss = KL_loss + KL_weight * self.beta_x * KL_x_t

        # -------------------  RL loss --------------------------------
        mask_batch = torch.unsqueeze(mask_batch, -1)
        maskp_batch = torch.unsqueeze(maskp_batch, -1)
        d_batch = torch.unsqueeze(d_batch, -1)
        r_batch = torch.unsqueeze(r_batch, -1)

        if self.algorithm == "sac":

            if isinstance(self.beta_h, str):
                beta_h = torch.exp(self.log_beta_h).detach()
            else:
                beta_h = self.beta_h

            ha_tensor, _ = self.rnn_a(h_q_tensor[:, :-1], torch.unsqueeze(ha_beg, 0))

            with torch.no_grad():
                mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
                siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

                sampled_u = self.sample_z(mua_tensor.detach(), siga_tensor.detach()).detach()
                sampled_a = torch.tanh(sampled_u)

                log_pi_exp = torch.sum(- (mua_tensor.detach() - sampled_u.detach()).pow(2)
                                        / (siga_tensor.detach().pow(2)) / 2
                                        - torch.log(siga_tensor.detach() * torch.tensor(2.5066)),
                                        dim=-1, keepdim=True)
                log_pi_exp = log_pi_exp - torch.sum(torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1,
                                                    keepdim=True)
                log_pi_exp = (log_pi_exp * mask_batch).detach().mean() / mask_batch.mean()

            # ------ loss_v ---------------
            input_v = self.f_x2phi_v(x_batch.reshape([minibatch_size * (max_stps + 1), *self.input_size])).reshape(
                [minibatch_size, max_stps + 1, -1])
            hv_tensor, _ = self.rnn_v(input_v, torch.unsqueeze(hv_beg, 0))
            
            hv_tensor_tar, _ = self.target_net.rnn_v(input_v, torch.unsqueeze(hv_tar_beg, 0))
            
            v_tensor = self.f_s2v(hv_tensor[:, :-1])
            vp_tensor = self.target_net.f_s2v(hv_tensor_tar[:, 1:]).detach()

            q_tensor_1 = self.f_sa2q1(hv_tensor[:, :-1], a_batch) 
            q_tensor_2 = self.f_sa2q2(hv_tensor[:, :-1], a_batch)

            sampled_q = torch.min(self.f_sa2q1(hv_tensor[:, :-1], sampled_a).detach(),
                                  self.f_sa2q2(hv_tensor[:, :-1], sampled_a).detach())

            q_exp = sampled_q

            v_tar = (q_exp - beta_h * log_pi_exp).detach() 

            loss_v = 0.5 * self.mse_loss(v_tensor * mask_batch, v_tar * mask_batch)

            loss_v = torch.mean(w_batch * loss_v.mean([1, 2]))

            loss_q = 0.5 * self.mse_loss(q_tensor_1 * mask_batch, (r_batch + (
                        1 - d_batch) * self.gamma * vp_tensor.detach()) * mask_batch) + \
                        0.5 * self.mse_loss(q_tensor_2 * mask_batch, (r_batch + (
                        1 - d_batch) * self.gamma * vp_tensor.detach()) * mask_batch)

            loss_q = torch.mean(w_batch * loss_q.mean([1, 2]))

            loss_critic = loss_q + loss_v

            loss = loss + loss_critic

            # ----- loss_a --------

            # Reparameterize a
            mua_tensor, logsia_tensor = self.f_s2pi0(ha_tensor)
            siga_tensor = torch.exp(logsia_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX))

            mu_prob = dis.Normal(mua_tensor, siga_tensor)
            sampled_u = mu_prob.rsample()
            sampled_a = torch.tanh(sampled_u)

            log_pi = torch.sum(mu_prob.log_prob(sampled_u).clamp(LOG_STD_MIN, LOG_STD_MAX), dim=-1,
                                    keepdim=True) - torch.sum(
                torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
            
            loss_a = torch.mean(w_batch * torch.mean(
                beta_h * log_pi * mask_batch - torch.min(
                    self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a),
                    self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a)
                ) * mask_batch + torch.min(
                    self.f_sa2q1(hv_tensor.detach()[:, :-1], sampled_a.detach()),
                    self.f_sa2q2(hv_tensor.detach()[:, :-1], sampled_a.detach())
                ) * mask_batch, dim=[1, 2]))

            loss_a = loss_a + REG / 2 * (
                torch.mean(w_batch * torch.mean((siga_tensor * mask_batch.repeat_interleave(
                    siga_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2]))
                + torch.mean(w_batch * torch.mean((mua_tensor * mask_batch.repeat_interleave(
                    mua_tensor.size()[-1], dim=-1)).pow(2), dim=[1, 2])))

            loss = loss + self.a_coef * loss_a

            # --------------------------------------------------------------------------

            # update entropy coefficient if required
            if isinstance(beta_h, torch.Tensor):
                self.optimizer_e.zero_grad()
                loss_e = torch.mean(- self.log_beta_h * (log_pi_exp + self.target_entropy).detach())
                loss_e.backward()
                self.optimizer_e.step()

            self.rl_update_times += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if hasattr(self, "anneal"):
                self.beta_h = self.beta_h * self.anneal

            # update target network                 
            state_dict_tar = self.target_net.f_s2v.state_dict()
            state_dict = self.f_s2v.state_dict()
            for key in list(state_dict_tar.keys()):
                state_dict_tar[key] = (1 - 0.005) * state_dict_tar[key] + 0.005 * state_dict[key]
            self.target_net.f_s2v.load_state_dict(state_dict_tar)

            state_dict_tar = self.target_net.rnn_v.state_dict()
            state_dict = self.rnn_v.state_dict()
            for key in list(state_dict_tar.keys()):
                state_dict_tar[key] = (1 - 0.005) * state_dict_tar[key] + 0.005 * state_dict[key]
            self.target_net.rnn_v.load_state_dict(state_dict_tar)

        else:
            raise NotImplementedError

        return KL_x.detach().cpu().item(), kld.detach().cpu().item()
