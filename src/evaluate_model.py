import argparse
import os
import torch
import wandb

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_fold', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
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
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.eval()
    return generator


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


def evaluate(args, loader, generator, num_samples):
    ade_out, fde_out, ade_std_out, fde_std_out = [],[],[],[]
    
    with torch.no_grad():
        for _ in range(10):
            ade_outer, fde_outer = [], []
            total_traj = 0
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end) = batch

                ade, fde = [], []
                ade_std, fde_std = [], []
                total_traj += pred_traj_gt.size(1)

                for _ in range(num_samples):
                    pred_traj_fake_rel = generator(
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

            ade = sum(ade_outer) / (total_traj * args.pred_len)
            fde = sum(fde_outer) / (total_traj)

            ade_out.append(ade.cpu())
            fde_out.append(fde.cpu())

    return ade_out, fde_out


def main(args):
    os.environ["WANDB_API_KEY"] = 'b098b7dbae4e9dd2520c5115fab0b6f8752ea865'

    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        try:
            checkpoint = torch.load(path)
        except:
            continue
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])


        wandb.init(
            project="trajgan_results_test", 
            config=_args, 
            tags=[_args.tag, _args.dataset_name],
            reinit=True,
            name=f'{_args.tag}_{_args.dataset_name}')

        eval_dict = {
            'ade': None,
            'fde': None,
            'ade_std': None,
            'fde_std': None
            }
        path = get_dset_path(
            os.path.join('/data/trajgan/datasets/datasets_real/', _args.dataset_name), 'test'
            )
        _, loader = data_loader(_args, path)

        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('- Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, np.mean(ade), np.mean(fde)))

        '''
        with open(os.path.join(args.checkpoint_fold, f'results_{_args.tag}.txt'), 'a') as f:
            f.write('\
                Dataset: {},\
                Pred Len: {},\
                ADE: {:.2f},\
                FDE: {:.2f},\
                sd(ade): {:.4f},\
                sd(fde): {:.4f}'.format(
                    _args.dataset_name, 
                    12, 
                    np.mean(ade), 
                    np.mean(fde),
                    np.std(ade),
                    np.std(fde)
                    )
                )
            f.write('\n')
            '''

        eval_dict['ade'] = np.mean(ade)
        eval_dict['fde'] = np.mean(fde)
        eval_dict['ade_std'] = np.std(ade)
        eval_dict['fde_std'] = np.std(fde)

        wandb.log({
                'results': eval_dict
            })




if __name__ == '__main__':
    args = parser.parse_args()
    for t in ['baseline']: #os.listdir(args.checkpoint_fold):
        args.model_path = os.path.join(args.checkpoint_fold, t)
        main(args)
