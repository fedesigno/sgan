import os
import math
import torch
import numpy as np
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)

def loc_pos_ts(seq_):

    # seq_ [obs_len N 2]
    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]

    pos_seq = torch.arange(1, obs_len + 1).cuda()
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat_interleave(num_ped, axis=1)
    result = torch.cat((pos_seq, seq_), axis=-1)
    return result

def loc_pos(seq_):

    # seq_ [obs_len N 2]

    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]

    pos_seq = np.arange(1, obs_len + 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_ped, axis=1)

    result = np.concatenate((pos_seq, seq_), axis=-1)

    return result



def seq_to_graph(seq_, seq_rel, pos_enc=False):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]

    if pos_enc:
        V = loc_pos(V)

    return torch.from_numpy(V).type(torch.float)

def seq_to_graph_ts(seq_, seq_rel, pos_enc=False):
    #seq_, seq_rel = seq_.cpu().numpy(), seq_rel.cpu().numpy()
    #seq_ = seq_.squeeze()
    #seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[0]
    max_nodes = seq_.shape[1]

    V = torch.zeros((seq_len, max_nodes, 2)).cuda()
    for s in range(seq_len):
        step_ = seq_[s, :, :]
        step_rel = seq_rel[s, :, :]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]

    if pos_enc:
        V = loc_pos_ts(V)

    return V



def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)
