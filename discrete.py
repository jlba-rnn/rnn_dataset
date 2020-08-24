import numpy as np
import gym
import pandas as pd
from sklearn.cluster import KMeans

class DiscretizeFeatures():
  def __init__(self, k=3, per_dim=True):
    self.kmeans = [] 
    self.k = k 
    self.per_dim = per_dim

  def fitKmeans(self, data,):
    if self.per_dim:
      design_matrix_s = np.vstack(np.array(data['s']))
      num_feat = design_matrix_s.shape[1]
      data_idx_s = []
      for i in range(num_feat):
        data_array_s = design_matrix_s[:, i].reshape(-1, 1)
        self.kmeans.append(KMeans(n_clusters=self.k, random_state=0).fit(data_array_s))
    else:
      design_matrix_s = np.vstack(np.array(data['s']))
      self.kmeans.append(KMeans(n_clusters=self.k, random_state=0).fit(design_matrix_s))

  def encode(self, feat,):
    if self.per_dim:
      num_feat = feat.shape[1]
      feat_int = []
      for i in range(num_feat):
        feat_dim_raw = feat[:, i].reshape(-1, 1)
        feat_int.append(self.kmeans[i].predict(feat_dim_raw).reshape(-1, 1))
      feat_int = np.hstack(feat_int)
    else:
      feat_int = self.kmeans[-1].predict(feat).reshape(-1, 1)
    return feat_int

  def decode(self, feat_int,):
    if self.per_dim:
      num_feat = feat_int.shape[1]
      feat = []
      for i in range(num_feat):
        feat_dim_int = feat_int[:, i].reshape(-1, 1)
        feat.append(self.kmeans[i].cluster_centers_[feat_dim_int].reshape(-1, 1))
      feat = np.hstack(feat)
    else:
      feat = np.array(self.kmeans[-1].cluster_centers_[feat_int]).squeeze()
    return feat

