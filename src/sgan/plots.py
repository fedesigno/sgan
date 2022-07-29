import matplotlib.pyplot as plt
import numpy as np
import os

def score_plot(args, real, gen, type_):
    plot_dir = "/scratch/sgcn_trajgan/src/plot"+type_
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    real = np.asarray(real)
    plt.plot(gen, 'b-', alpha=0.9)
    plt.plot(real, 'y-', alpha=0.7)

    plt.legend([type_+' generated', type_+' real'])#, type_+' synthetic'])
    plt.ylabel(type_)
    plt.xlabel('epochs')
    #plt.ylim(0, 2)
    plt.title(type_+' by epoch')
    plt.savefig(plot_dir+"/"+type_+".png")
    plt.close()

def traj_plot(args, batch, title, k):
    if k==0:
        c='y.'
    elif k==1:
        c='g.'
    elif k==2:
        c='b.'

    plot_dir = "/scratch/sgan/src/plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    x = batch[:,:,0].cpu()
    y = batch[:,:,1].cpu()
    plt.plot(x.detach().numpy(), y.detach().numpy(), c, alpha=0.4)
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'k-', alpha=0.2)
    plt.title('trajectories-'+title)
    plt.savefig(plot_dir+"/"+title+".png")
    plt.close()


def mse_plot(args, loss, type_='mse'):
    plot_dir = "/scratch/social-adapt-gan/sgan/plots/"+args.version
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    loss = np.asarray(loss)
    plt.plot(loss, 'm-', alpha=0.9)
    plt.legend([type_+' mse'])
    plt.ylabel(type_)
    plt.xlabel('epochs')
    #plt.ylim(0, 2)
    plt.title(type_+' by epoch')
    plt.savefig(plot_dir+"/"+args.dataset_name+"_"+type_+".png")
    plt.close()

def plot_results(args, losses_dict):
    plot_dir = "/scratch/social-adapt-gan/sgan/plots/"+args.version+"/"+args.dataset_name
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_trend(
        args,
        np.asarray(losses_dict['loss_d']['real']),
        np.asarray(losses_dict['loss_d']['gen']),
        plot_dir,  type_ = 'discriminator'
        )

    plot_trend(
        args,
        np.asarray(losses_dict['loss_p']['ade_real']),
        np.asarray(losses_dict['loss_p']['ade_gen']),
        plot_dir,  type_ = 'predictor_ade'
        )

    plot_trend(
        args,
        np.asarray(losses_dict['loss_p']['fde_real']),
        np.asarray(losses_dict['loss_p']['fde_gen']),
        plot_dir,  type_ = 'predictor_fde'
        )

    plot_trend(
        args,
        np.asarray(losses_dict['loss_p']['disc_real']),
        np.asarray(losses_dict['loss_p']['disc_gen']),
        plot_dir,  type_ = 'predictor_bce'
        )

    plot_trend_gen(
        args,
        np.asarray(losses_dict['loss_g']),
        plot_dir,  type_ = 'generator'
        )
    
def plot_trend(args, real, gen, plot_dir, type_):

    plt.plot(gen, 'b-', alpha=0.9)
    plt.plot(real, 'y-', alpha=0.7)

    plt.legend([type_+' generated', type_+' real'])#, type_+' synthetic'])
    plt.ylabel(type_)
    plt.xlabel('epochs')
    #plt.ylim(0, 2)
    plt.title(type_+' by epoch')
    plt.savefig(plot_dir+"/"+type_+"_"+args.dataset_name+".png")
    plt.close()

def plot_trend_gen(args, gen, plot_dir, type_):
    plt.plot(gen, 'b-', alpha=0.9)
    plt.legend([type_+' generated'])#, type_+' synthetic'])
    plt.ylabel(type_)
    plt.xlabel('epochs')
    #plt.ylim(0, 2)
    plt.title(type_+' by epoch')
    plt.savefig(plot_dir+"/"+type_+"_"+args.dataset_name+".png")
    plt.close()