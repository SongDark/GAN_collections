# coding:utf-8
'''
Ref:
https://github.com/hwalsuklee/tensorflow-generative-model-collections
https://github.com/pfnet-research/sngan_projection
https://github.com/taki0112/Spectral_Normalization-Tensorflow
https://github.com/shekkizh/WassersteinGAN.tensorflow
https://github.com/handspeaker/gan_practice
'''
import argparse, os, glob
from gan import GAN
from cgan import CGAN
from infogan import InfoGAN

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def check_args(args):
    # check_folder(args.checkpoint_dir)
    # check_folder(args.fig_dir)
    # check_folder(args.log_dir)
    assert args.epoch >= 1, 'non-positive epoches'
    assert args.batch_size >= 1, 'non-positive batch_size'
    assert args.z_dim >= 1, 'non-positive noise dimension'
    return args

def parse_args():
    desc = "Tensorflow implementation of GANs"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--todo', type=str, default='train',
                        choices=['train','clear','clearanyway'])
    parser.add_argument('--gan_type', type=str, default='gan', 
                        choices=['gan', 'wgan', 'wgan_gp', 'lsgan', 'dragan', 'cgan', 'infogan'])
    parser.add_argument('--net_type', type=str, default='cnn', 
                        choices=['cnn', 'mlp'])
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist',])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=50)
    parser.add_argument('--clip', type=float, default=0.05)
    parser.add_argument('--D_iter', type=int, default=5)
    parser.add_argument('--plot_iter', type=int, default=5)
    parser.add_argument('--verbose', type=bool, default=True)
    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    # parser.add_argument('--fig_dir', type=str, default='figs')
    # parser.add_argument('--log_dir', type=str, default='logs')

    return check_args(parser.parse_args())

def main():
    args = parse_args()
    if args is None:
        exit()

    if 'clear' in args.todo:
        version = args.gan_type + "_" + args.net_type
        logs_dir = os.path.join('logs', version)
        figs_dir = os.path.join('figs', version)
        mods_dir = os.path.join('checkpoints', version)
        
        logs = glob.glob(os.path.join(logs_dir, "*"))
        figs = glob.glob(os.path.join(figs_dir, "*"))
        mods = glob.glob(os.path.join(mods_dir, "*"))
        
        if "anyway" in args.todo:
            print version, "cleared."
            for f in logs+figs+mods:
                os.remove(f)
            exit()
        
        print "You will delete files of:", version
        order = raw_input("[Delete or not? (y/n)] ")
        if order == 'y' or order == 'Y':
            for f in logs+figs+mods:
                os.remove(f)
        exit()
    
    elif args.todo == 'train':
        if args.gan_type == 'cgan':
            gan = CGAN(batch_size=args.batch_size,
                       net_type=args.net_type,
                       noise_dim=args.z_dim,
                       critic_iter=args.D_iter)
        elif args.gan_type == 'infogan':
            gan = InfoGAN(batch_size=args.batch_size,
                          net_type=args.net_type,
                          noise_dim=args.z_dim,
                          critic_iter=args.D_iter,
                          plot_iter=args.plot_iter,
                          verbose=args.verbose)
        else:
            gan = GAN(gan_type=args.gan_type,
                    net_type=args.net_type,
                    batch_size=args.batch_size,
                    noise_dim=args.z_dim,
                    clip_num=args.clip,
                    critic_iter=args.D_iter,
                    plot_iter=args.plot_iter,
                    verbose=args.verbose)
        gan.train(epoches=args.epoch)

if __name__ == '__main__':
    main()
    




# python main.py --gan_type gan --net_type cnn --epoch 30 --clip 0 --D_iter 1
# python main.py --gan_type gan --net_type mlp --epoch 30 --clip 0 --D_iter 5
# python main.py --gan_type wgan --net_type cnn --epoch 30 --clip 0.05 --D_iter 5
# python main.py --gan_type wgan --net_type mlp --epoch 30 --clip 0.1 --D_iter 5
# python main.py --gan_type wgan_gp --net_type cnn --epoch 30 --clip 0 --D_iter 1
# python main.py --gan_type wgan_gp --net_type mlp --epoch 30 --clip 0 --D_iter 1
# python main.py --gan_type lsgan --net_type cnn --epoch 30 --clip 0.05 --D_iter 1 # must gc
# python main.py --gan_type lsgan --net_type mlp --epoch 30 --clip 0.05 --D_iter 1
# python main.py --gan_type dragan --net_type cnn --epoch 30 --clip 0 --D_iter 1
# python main.py --gan_type dragan --net_type mlp --epoch 30 --clip 0 --D_iter 5

# python main.py --gan_type cgan --net_type cnn --epoch 30 --D_iter 1
# python main.py --gan_type cgan --net_type mlp --epoch 30 --D_iter 1

# python main.py --gan_type infogan --net_type cnn --epoch 30 --D_iter 1
# python main.py --gan_type infogan --net_type mlp --epoch 30 --D_iter 1