import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

#%% check some of the trajectories
def plotTraj_pd(data, dim1 = 0, dim2 = 2, data_pred =None):
  f,ax = plt.subplots(1,2, figsize=(20,10));
  # sample trajectories
  for i in range(10):    
      df_ep = data.query('episode=='+str(i))
      line, = ax[0].plot(df_ep['s'].apply(lambda x:x[dim1]),
                df_ep['s'].apply(lambda x:x[dim2])*180/3.14, alpha = 0.3)
      ax[0].plot(df_ep['s'].apply(lambda x:x[dim1]).iloc[0],
                df_ep['s'].apply(lambda x:x[dim2]).iloc[0]*180/3.14, 
              color = line.get_color(), marker = 'o',  alpha = 0.5)
      if data_pred is not None:
        df_ep = data_pred.query('episode=='+str(i))
        line, = ax[0].plot(df_ep['s'].apply(lambda x:x[dim1]),
                  df_ep['s'].apply(lambda x:x[dim2])*180/3.14, ':', alpha = 0.3)
        ax[0].plot(df_ep['s'].apply(lambda x:x[dim1]).iloc[0],
                  df_ep['s'].apply(lambda x:x[dim2]).iloc[0]*180/3.14, 
                color = line.get_color(), marker = 'o',  alpha = 0.5)
  ax[0].set_xlabel('cart position'); ax[0].set_ylabel('pole angle (deg)')

  # check input dist.
  ax[1].boxplot(np.asarray(list(data['s'])))


#%% check some of the trajectories
def plotTraj(data, dim1 = 0, dim2 = 2, data_pred =None):
  f,ax = plt.subplots(1,2, figsize=(20,10));
  # sample trajectories
  for i in range(10):    
      df_ep = data[i]
      line, = ax[0].plot(df_ep[:, dim1], 
                df_ep[:, dim2]*180/3.14, alpha = 0.3)
      ax[0].plot(df_ep[0, dim1], 
                df_ep[0, dim2]*180/3.14, 
              color = line.get_color(), marker = 'o',  alpha = 0.5)
      if data_pred is not None:
        df_ep = data_pred[i]
        line, = ax[0].plot(df_ep[:, dim1], 
                df_ep[:, dim2]*180/3.14, ':', alpha = 0.3)
        ax[0].plot(df_ep[0, dim1], 
                  df_ep[0, dim2]*180/3.14, 
                color = line.get_color(), marker = 'o',  alpha = 0.5)
  ax[0].set_xlabel('cart position'); ax[0].set_ylabel('pole angle (deg)')

def visualizeTraj(model,):
  ## multi-step prediction validation
  state_next_pred = multi_pred(model, [state_valid, action_valid])
  state_next_pred *= len_masks_valid
  for i in [0, 1, 2]:
    plt.figure()
    plt.plot(state_next_valid[i,:,:])
    plt.plot(state_next_pred[i,:,:], '--')


