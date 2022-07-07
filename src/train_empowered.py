import argparse
from ast import arg
import gc
import logging
import os
import sys
import time
import wandb

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.cuda.amp.GradScaler as GradScaler

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, gan_d_g_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--dataset_dir', default='/data', type=str)
parser.add_argument('--dataset_dir_synth', default='/data', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--tag', default="baseline", type=str)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# predictor Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=1e-3, type=float)
parser.add_argument('--g_steps', default=1, type=int)

parser.add_argument('--p_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=1e-3, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default='/scratch/sgan/src/checkpoint')
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    train_path = get_dset_path(
        os.path.join(
            args.dataset_dir, 
            args.dataset_name), 
            f'train_{args.dataset_name}'
            )
    val_path = get_dset_path(
        os.path.join(
            args.dataset_dir, 
            args.dataset_name), 'test'
            )

    jta_path = get_dset_path(
        os.path.join(
            args.dataset_dir_synth, 
            args.dataset_name), 
            f'train_{args.dataset_name}'
            )

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing jta dataset")
    jta_dset, jta_loader = data_loader(args, jta_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    predictor = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    predictor.apply(init_weights)
    predictor.type(float_dtype).train()
    logger.info('Here is the predictor:')
    logger.info(predictor)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    generator = TrajectoryGenerator(
        obs_len=20,
        pred_len=20,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the predictor:')
    logger.info(generator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss
    d_loss_g_fn = gan_d_g_loss

    optimizer_p = optim.Adam(
        predictor.parameters(), lr=args.g_learning_rate
        )
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )
    optimizer_g = optim.Adam(
        generator.parameters(), lr=args.g_learning_rate
    )
    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        predictor.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_p.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'P_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'metrics_jta': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'norm_p': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'p_state': None,
            'p_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'p_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'p_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    
    eval_dict = {'min_ade': None, 'min_fde': None}
    t0 = None
    scaler = torch.cuda.amp.GradScaler() #GradScaler()
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        p_steps_left = args.p_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in zip(train_loader, jta_loader):
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # predictor; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the predictor.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, predictor,
                                              discriminator, generator,
                                              d_loss_fn, d_loss_g_fn,
                                              optimizer_d, scaler)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif p_steps_left > 0:
                step_type = 'p'
                losses_p = predictor_step(args, batch, predictor,
                                          discriminator, generator,
                                          g_loss_fn,
                                          optimizer_p, scaler)
                checkpoint['norm_p'].append(
                    get_total_norm(predictor.parameters())
                )
                p_steps_left -= 1


            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch[1], generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g, scaler)
                checkpoint['norm_g'].append(
                    get_total_norm(predictor.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0 or p_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)

                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
  
                for k, v in sorted(losses_p.items()):
                    logger.info('  [P] {}: {:.3f}'.format(k, v))
                    checkpoint['P_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, predictor, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, predictor, discriminator,
                    d_loss_fn, limit=True
                )
                metrics_jta = check_accuracy(
                    args, jta_loader, predictor, discriminator,
                    d_loss_fn, limit=True
                )
                wandb.log({
                        'train': metrics_train, 
                        'val': metrics_val, 
                        'jta': metrics_jta,
                        'epoch': epoch,
                        't': t
                    })

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [jta] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_jta'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_fde = min(checkpoint['metrics_val']['fde'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = predictor.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()


                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['p_state'] = predictor.state_dict()
                checkpoint['p_optim_state'] = optimizer_p.state_dict()
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()

                eval_dict['min_ade'] = min_ade
                eval_dict['min_fde'] = min_fde

                wandb.log({
                        'train': metrics_train, 
                        'val': metrics_val, 
                        'best': eval_dict,
                        'epoch': epoch,
                        't': t
                    })

                
                out_path = os.path.join(args.output_dir, args.tag)
                out_name = args.dataset_name +'_'+args.tag
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                checkpoint_path = os.path.join(
                    out_path, '%s_with_model.pt' % out_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    out_path, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break

def discriminator_step(
    args, batch_, predictor, discriminator, generator, d_loss_fn, d_loss_g_fn, optimizer_d, scaler
):
    for i, batch in enumerate(batch_):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
            loss_mask, seq_start_end) = batch

        if i==1:
            traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            generator_out = generator(traj, traj_rel, seq_start_end)
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])  
            obs_traj_rel, pred_traj_gt_rel = torch.split(pred_traj_fake_rel, [8,12])
            obs_traj, pred_traj_gt = torch.split(pred_traj_fake, [8,12])

        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)
        
        #with torch.cuda.amp.autocast():
        predictor_out = predictor(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = predictor_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

        # Compute loss with optional gradient penalty
        if i==1:
            data_loss = d_loss_g_fn(scores_real, scores_fake, 'fake')
        else:
            data_loss = d_loss_g_fn(scores_real, scores_fake, 'real')
            #data_loss = d_loss_fn(scores_real, scores_fake)

        #scaler.scale(loss).backward()
        #scaler.unscale_(optimizer_d)

        losses['D_data_loss'] = data_loss.item()
        loss += data_loss
        losses['D_total_loss'] = loss.item()
        optimizer_d.zero_grad()
        loss.backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(),
                                        args.clipping_threshold_d)
        optimizer_d.step()
        #scaler.step(optimizer_d)
        #scaler.update()

    return losses

def predictor_step(
args, batch_, predictor, discriminator, generator, g_loss_fn, optimizer_p, scaler
):
    for i, batch in enumerate(batch_):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
            loss_mask, seq_start_end) = batch

        if i==1:
            traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            generator_out = generator(traj, traj_rel, seq_start_end)
            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])  
            obs_traj_rel, pred_traj_gt_rel = torch.split(pred_traj_fake_rel, [8,12])
            obs_traj, pred_traj_gt = torch.split(pred_traj_fake, [8,12])

        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)
        g_l2_loss_rel = []

        loss_mask = loss_mask[:, args.obs_len:]

        for _ in range(args.best_k):
            predictor_out = predictor(obs_traj, obs_traj_rel, seq_start_end)

            pred_traj_fake_rel = predictor_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            if args.l2_loss_weight > 0:
                g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                    pred_traj_fake_rel,
                    pred_traj_gt_rel,
                    loss_mask,
                    mode='raw'))

        g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        if args.l2_loss_weight > 0:
            g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _g_l2_loss_rel = g_l2_loss_rel[start:end]
                _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                    loss_mask[start:end])
                g_l2_loss_sum_rel += _g_l2_loss_rel
            losses['P_l2_loss_rel'] = g_l2_loss_sum_rel.item()
            loss += g_l2_loss_sum_rel

        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        discriminator_loss = g_loss_fn(scores_fake)

        loss += discriminator_loss
        #scaler.scale(loss).backward()
        #scaler.unscale_(optimizer_p)
        losses['P_discriminator_loss'] = discriminator_loss.item()
        losses['P_total_loss'] = loss.item()

        optimizer_p.zero_grad()
        loss.backward()
        if args.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                predictor.parameters(), args.clipping_threshold_g
            )
        #scaler.step(optimizer_p)
        optimizer_p.step()
        #scaler.update()

        return losses

def generator_step(
args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
        loss_mask, seq_start_end) = batch
    
    obs_traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
    obs_traj_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

    losses = {}
    loss = torch.zeros(1).to(obs_traj)
    g_l2_loss_rel = []

    #loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(1): #args.best_k):
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

    return losses

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

def check_accuracy(
    args, loader, predictor, discriminator, d_loss_fn, limit=False):

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
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end) = batch

            loss_mask = loss_mask[:, args.obs_len:]

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

    
            pred_traj_fake_rel = predictor(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )

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


            loss_mask_sum += torch.numel(loss_mask.data)

            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

        ade_ = sum(ade_outer) / (total_traj * args.pred_len)
        fde_ = sum(fde_outer) / (total_traj)

        metrics['d_loss'] = sum(d_losses) / len(d_losses)
        metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
        metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

        metrics['ade'] = ade_.cpu()
        metrics['fde'] = fde_.cpu()

        predictor.train()

    return metrics

def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel

def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl

def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = 'b098b7dbae4e9dd2520c5115fab0b6f8752ea865'
    for dataset_name in ['eth']: #, 'hotel', 'univ', 'zara1', 'zara2']:
        args.dataset_name = dataset_name
        wandb.init(
            project="traj-generation", 
            config=args, 
            tags=[args.tag, args.dataset_name],
            name=f'{args.tag}_{args.dataset_name}')
        main(args)
