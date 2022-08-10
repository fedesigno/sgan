import os
from re import T
import time
import torch
import torch.nn as nn
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess

from sgan.losses import cal_l2_losses, l2_loss, displacement_error, final_displacement_error
from sgcn.utils import seq_to_graph
from sgcn.metrics import nodes_rel_to_nodes_abs_np, seq_to_nodes, nodes_rel_to_nodes_abs, ade, fde

import torch.distributions.multivariate_normal as torchdist
import copy



def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


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



def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def check_accuracy_(
    args, loader, predictor, discriminator, d_loss_fn, t, epoch, limit=False):

    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    ade_outer, fde_outer = [], []

    loss_mask_sum = 0
    predictor.eval()

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr, seq_start_end = batch

            loss_mask = loss_mask[:, args.obs_len:]

            #ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                            torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
            identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda') * \
                                torch.eye(V_obs.shape[1], device='cuda')  # [N obs_len obs_len]
            identity = [identity_spatial, identity_temporal]

            V_pred = predictor(V_obs, identity)  # A_obs <8, #, #>
            V_pred = V_pred.squeeze()
            V_tr = V_tr.squeeze()

            V_x = obs_traj #seq_to_nodes(obs_traj_, n) 
            V_rel_to_abs = nodes_rel_to_nodes_abs(V_pred[:,:,:2], 
                                                    V_x[-1,:,:]) 

            pred_traj_fake = V_rel_to_abs
            pred_traj_fake_rel = V_pred[:,:,:2]

            ade.append(displacement_error(
                pred_traj_fake, pred_traj_gt, mode='raw'
            ))
            fde.append(final_displacement_error(
                pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
            ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

            '''
            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())

            '''
            loss_mask_sum += torch.numel(loss_mask.data)

            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break
            

        ade_ = sum(ade_outer) / (total_traj * args.pred_len)
        fde_ = sum(fde_outer) / (total_traj)

        #metrics['d_loss'] = sum(d_losses) / len(d_losses)
        #metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
        #metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

        metrics['ade'] = ade_.cpu()
        metrics['fde'] = fde_.cpu()
        metrics['t'] = t 
        metrics['epoch'] = epoch

        predictor.train()

    return metrics

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)




def check_accuracy(
    args, loader, predictor, discriminator, d_loss_fn, t, epoch, limit=False):
    raw_data_dict = {}
    ade_bigls = []
    fde_bigls = []

    step =0
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    ade_outer, fde_outer = [], []

    loss_mask_sum = 0
    predictor.eval()

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr, seq_start_end = batch

            loss_mask = loss_mask[:, args.obs_len:]

            #ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                            torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
            identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda') * \
                                torch.eye(V_obs.shape[1], device='cuda')  # [N obs_len obs_len]
            identity = [identity_spatial, identity_temporal]

            V_pred = predictor(V_obs, identity)  # A_obs <8, #, #>
            V_pred = V_pred.squeeze()
            V_tr = V_tr.squeeze()

            num_of_objs = obs_traj_rel.shape[1]
            V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
            #
            # #For now I have my bi-variate parameters
            # #normx =  V_pred[:,:,0:1]
            # #normy =  V_pred[:,:,1:2]
            sx = torch.exp(V_pred[:,:,2]) #sx
            sy = torch.exp(V_pred[:,:,3]) #sy
            corr = torch.tanh(V_pred[:,:,4]) #corr
            #
            cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
            cov[:,:,0,0]= sx*sx
            cov[:,:,0,1]= corr*sx*sy
            cov[:,:,1,0]= corr*sx*sy
            cov[:,:,1,1]= sy*sy
            mean = V_pred[:,:,0:2]

            mvnormal = torchdist.MultivariateNormal(mean,cov)
            #
            #
            # ### Rel to abs
            # ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len
            #
            # #Now sample 20 samples
            ade_ls = {}
            fde_ls = {}
            V_x = obs_traj.squeeze(0).cpu().numpy().copy() #seq_to_nodes(obs_traj.cpu().numpy().copy())
            V_x_rel_to_abs = nodes_rel_to_nodes_abs_np(V_obs[:,:,:,:2].cpu().numpy().squeeze().copy(),
                                                    V_x[0,:,:].copy())
            #
            #V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
            V_y_rel_to_abs = nodes_rel_to_nodes_abs_np(V_tr.cpu().numpy().squeeze().copy(),
                                                    V_x[-1,:,:].copy())

            raw_data_dict[step] = {}
            raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
            raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
            raw_data_dict[step]['pred'] = []
            #
            #
            for n in range(num_of_objs):
                ade_ls[n]=[]
                fde_ls[n]=[]
            #
            for k in range(10):

                V_pred = mvnormal.sample()

                V_pred_rel_to_abs = nodes_rel_to_nodes_abs_np(V_pred.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1,:,:].copy())

                raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))


                for n in range(num_of_objs):
                    pred = []
                    target = []
                    obsrvs = []
                    number_of = []
                    pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                    target.append(V_y_rel_to_abs[:,n:n+1,:])
                    obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                    number_of.append(1)
            #
                    ade_ls[n].append(ade(pred,target,number_of))
                    fde_ls[n].append(fde(pred,target,number_of))

            for n in range(num_of_objs):
                ade_bigls.append(min(ade_ls[n]))
                fde_bigls.append(min(fde_ls[n]))

        ade_ = sum(ade_bigls)/len(ade_bigls)
        fde_ = sum(fde_bigls)/len(fde_bigls)

        metrics['ade'] = ade_
        metrics['fde'] = fde_
        metrics['t'] = t 
        metrics['epoch'] = epoch

        predictor.train()

        return metrics
