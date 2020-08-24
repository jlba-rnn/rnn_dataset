import numpy as np
import gym
import pandas as pd


def make_dataset(seed=1234, env_name='CartPole-v1', policy_type='random', 
                 normalize_data=True, num_traj=4000, MAX_LEN = 60):
  data = trajectories(seed, env_name, policy_type, num_traj)
  if normalize_data:
    data = normalize_traj(data)
  #%% make arrays padded to MAX_LEN for RNN 
  state_train, action_train, state_next_train, rwd_train, len_masks_train = make_array(data, MAX_LEN)
  num_train = num_traj // 4 * 3
  num_test = len(state_train) - num_train

  state_train, state_valid = state_train[:num_train], state_train[num_train:]
  action_train, action_valid = action_train[:num_train], action_train[num_train:]
  state_next_train, state_next_valid = state_next_train[:num_train], state_next_train[num_train:]
  len_masks_train, len_masks_valid = len_masks_train[:num_train], len_masks_train[num_train:]
  
  train_set = {'state':state_train, 'action':action_train, 'state_next':state_next_train, 'seq_mask':len_mask_train}
  valid_set = {'state':state_valid, 'action':action_valid, 'state_next':state_next_valid, 'seq_mask':len_mask_valid}
  return train_set, valid_set


def trajectories(seed, env_name, policy_type, num_traj):
  env = gym.make(env_name)
  
  env.seed(seed)
  env.action_space.np_random.seed(seed)
  np.random.seed(seed)
  # from gym.wrappers import Monitor
  # env = Monitor(env, './video')
  print('Stochastic? ', env.spec.nondeterministic)
  
  policy = Policy(env, policy_type)
  
  # collect trajectories
  data = collect_trajectories(env = env, policy = policy, N = num_traj) 
  data = pd.DataFrame(data, columns = ['episode', 'step', 's', 'a', 's_next', 'rwd', 'done'])

  return data



def normalize_traj(data):
  #%% normalize data
  state_mean = np.asarray(list(data['s'])).mean(axis=0)
  state_std = np.asarray(list(data['s'])).std(axis=0)

  #print(state_mean, state_std)

  data['s'] = data['s'].apply(lambda x:(x-state_mean)/state_std)    
  data['s_next'] = data['s_next'].apply(lambda x:(x-state_mean)/state_std)  
  return data


class Policy:
    def __init__(self, env, policy='random'):
        self.env = env
        self.policy = policy
        self.cnt = 0
        
    def sample(self, state):
        # sample one action from pi(s)
        if self.policy == 'random':
            return self.env.action_space.sample()   # random policy; uses open_gym API
        elif self.policy == 'always_0':
            return 0
        elif self.policy == 'always_1':
            return 1
        elif self.policy == 'alternate':
            self.cnt += 1
            return (self.cnt % 2)
        elif self.policy == 'heuristic':
            if state[3]<0:
                return 0
            else:
                return 1
        else:
            raise ValueError('unknwon policy')

def make_array(data, MAX_LEN):
    # output: padded arrays
    
    eps = data.episode.unique()
    State_dim = data.iloc[0]['s'].shape[0]
 
    def get_ep(ep, col, upto = None):
        if upto is None:
            return np.asarray(list(data.query('episode=='+str(ep))[col])).astype(np.float32)
        else:
            return np.asarray(list(data.query('episode=='+str(ep)+' and step<'+str(upto))[col])).astype(np.float32)
    
    states = np.zeros((len(eps), MAX_LEN, State_dim), dtype=np.float32)
    actions = np.zeros((len(eps), MAX_LEN, 1), dtype=np.float32)
    state_nexts = np.zeros((len(eps), MAX_LEN, State_dim), dtype=np.float32)
    rwds = np.zeros((len(eps), MAX_LEN, 1), dtype=np.float32)
    len_masks = np.zeros((len(eps), MAX_LEN, 1), dtype=np.float32)    
        
    for idx, ep in enumerate(eps):
        _data = get_ep(ep, 's', MAX_LEN)
        states[idx, :_data.shape[0], :] = _data
        actions[idx, :_data.shape[0], 0] = get_ep(ep, 'a', MAX_LEN)
        state_nexts[idx, :_data.shape[0], :] = get_ep(ep, 's_next', MAX_LEN)
        rwds[idx, :_data.shape[0], 0] = get_ep(ep, 'rwd', MAX_LEN)
        len_masks[idx, :_data.shape[0], 0] = 1.
            
    return states, actions, state_nexts, rwds, len_masks       

def collect_trajectories(env, policy, N, max_step_per_episode = 200, fix_seed_action = False, render = False):
    
    data = []
    for i in range(N):      # for N episodes
        step = 0
        if fix_seed_action:
          env.seed(0);       # for fixed initial position 
        s = env.reset(); 
        while (step < max_step_per_episode):
            if render:
              env.render()   # TODO: this doesn't work w. notebook
            if fix_seed_action:
              action = 0
            else:
              action = policy.sample(s) 
            s_next, reward, done, info = env.step(action)
            data.append([i, step, s, action, s_next, reward, done])
            s = s_next
            step += 1
            
            if done:
                break
        if ((i+1) % 100) == 0:
            print ("Collected %d trajectories"%(i+1) )
    return data
