import torch
import torch.nn as nn
from sgan.utils import relative_to_abs
from sgan.losses import l2_loss
from sgcn.utils import seq_to_graph, seq_to_graph_ts
from sgcn.metrics import graph_loss, seq_to_nodes, nodes_rel_to_nodes_abs


import matplotlib.pyplot as plt
import os
import wandb

def traj_plot(args, batch, title, k):
    if k==0:
        c='y.'
    elif k==1:
        c='g.'
    elif k==2:
        c='b.'

    x = batch[:,:,0].cpu()
    y = batch[:,:,1].cpu()
    plt.plot(x.detach().numpy(), y.detach().numpy(), c, alpha=0.4)
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'k-', alpha=0.2)
    plt.title('trajectories-'+title)
    wandb.log({f"{title}": plt})

    plt.close()



def discriminator_step(
    args, batch_, predictor, discriminator, generator, d_loss_fn, d_loss_g_fn, optimizer_d, t, epoch
):
    for i, batch in enumerate(batch_):
        batch = [tensor.cuda() for tensor in batch]
        #(obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
            #loss_mask, seq_start_end) = batch
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr, seq_start_end = batch
        
        if i==1:
            traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            generator_out = generator(traj, traj_rel, seq_start_end)
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])  
            obs_traj_rel, pred_traj_gt_rel = torch.split(pred_traj_fake_rel, [8,12])
            obs_traj, pred_traj_gt = torch.split(pred_traj_fake, [8,12])

            V_obs = seq_to_graph_ts(
                obs_traj,#.permute(1, 2, 0), 
                obs_traj_rel, 
                True
                ).unsqueeze(0)
            V_tr = seq_to_graph_ts(
                pred_traj_gt, 
                pred_traj_gt_rel, 
                False
                ).unsqueeze(0)
            type_ = 'jta'
        else:
            type_ = 'real'




        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                           torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda') * \
                            torch.eye(V_obs.shape[1], device='cuda')  # [N obs_len obs_len]

        identity = [identity_spatial, identity_temporal]
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        V_pred = predictor(V_obs, identity)  # A_obs <8, #, #>

        V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()

        V_x = obs_traj #seq_to_nodes(obs_traj_, n) #.cpu().numpy().copy(), n)
        V_rel_to_abs = relative_to_abs(V_pred[:,:,:2], #.detach().cpu().numpy().copy(),
                                                 V_x[-1]) #.copy())


        pred_traj_fake = V_rel_to_abs
        pred_traj_fake_rel = V_pred[:,:,:2]
        
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        traj_plot(args, traj_real, type_+"_from_gt", 0)
        traj_plot(args, traj_fake, type_+"_from_pred", 1)



        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

        # Compute loss with optional gradient penalty
        # if data are from predictor, than they're set all as 'fake'
        if i==1:
            data_loss = d_loss_g_fn(scores_real, scores_fake, 'fake')
        else:
            data_loss = d_loss_fn(scores_real, scores_fake)

        losses['D_data_loss'] = data_loss.item()
        loss += data_loss
        losses['D_total_loss'] = loss.item()
        optimizer_d.zero_grad()
        loss.backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(),
                                        args.clipping_threshold_d)
        optimizer_d.step()

        losses['t'] = t
        losses['epoch'] = epoch

    return losses


def predictor_step(
args, batch_, predictor, discriminator, generator, g_loss_fn, optimizer_p, last_iter, t, epoch
):
    for i, batch in enumerate(batch_):
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr, seq_start_end = batch

        if i==1:
            traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            generator_out = generator(traj, traj_rel, seq_start_end)
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])  
            obs_traj_rel, pred_traj_gt_rel = torch.split(pred_traj_fake_rel, [8,12])
            obs_traj, pred_traj_gt = torch.split(pred_traj_fake, [8,12])

            V_obs = seq_to_graph_ts(
                obs_traj,#.permute(1, 2, 0), 
                obs_traj_rel, 
                True
                ).unsqueeze(0)
            V_tr = seq_to_graph_ts(
                pred_traj_gt, 
                pred_traj_gt_rel, 
                False
                ).unsqueeze(0)

        optimizer_p.zero_grad()

        losses = {}

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                           torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda') * \
                            torch.eye(V_obs.shape[1], device='cuda')  # [N obs_len obs_len]

        identity = [identity_spatial, identity_temporal]
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        V_pred = predictor(V_obs, identity)  # A_obs <8, #, #>

        V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
 
        V_rel_to_abs = relative_to_abs(V_pred[:,:,:2], obs_traj[-1]) 


        pred_traj_fake = V_rel_to_abs
        pred_traj_fake_rel = V_pred[:,:,:2]


        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_plot(args, traj_fake, '_from_pred', 0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        discriminator_loss = g_loss_fn(scores_fake)

        loss += discriminator_loss
        losses['P_discriminator_loss'] = discriminator_loss.item()
 
        l = graph_loss(V_pred, V_tr)
        losses['P_graph_loss'] = l.item()

        loss+=l

        losses['P_total_loss'] = loss.item()

        #if last_iter:
        loss.backward()
        if args.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                predictor.parameters(), 10
            )
        optimizer_p.step()
        losses['t'] = t
        losses['epoch'] = epoch
        return losses


def generator_step(
args, batch, generator, discriminator, g_loss_fn, optimizer_g, t, epoch
):
    batch = [tensor.cuda() for tensor in batch]
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr, seq_start_end = batch
    
    obs_traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
    obs_traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

    losses = {}
    loss = torch.zeros(1).to(obs_traj)
    g_l2_loss_rel = []

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    if args.l2_loss_weight > 0:
        g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
            pred_traj_fake_rel,
            obs_traj_rel,
            loss_mask,
            mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(obs_traj)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = pred_traj_fake 
    traj_fake_rel = pred_traj_fake_rel 

    #traj_plot(traj_fake, 'fake_from_gen', 1)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss

    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()
    losses['t'] = t
    losses['epoch'] = epoch
    return losses