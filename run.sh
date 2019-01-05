echo "clear all"
python main.py --gan_type gan --net_type cnn --todo clearanyway 
python main.py --gan_type gan --net_type mlp --todo clearanyway
python main.py --gan_type wgan --net_type cnn --todo clearanyway
python main.py --gan_type wgan --net_type mlp --todo clearanyway
python main.py --gan_type wgan_gp --net_type cnn --todo clearanyway
python main.py --gan_type wgan_gp --net_type mlp --todo clearanyway
python main.py --gan_type dragan --net_type cnn --todo clearanyway
python main.py --gan_type dragan --net_type mlp --todo clearanyway
python main.py --gan_type lsgan --net_type cnn --todo clearanyway
python main.py --gan_type lsgan --net_type mlp --todo clearanyway
python main.py --gan_type cgan --net_type cnn --todo clearanyway
python main.py --gan_type cgan --net_type mlp --todo clearanyway
python main.py --gan_type infogan --net_type cnn --todo clearanyway
python main.py --gan_type infogan --net_type mlp --todo clearanyway
echo "train gan"
python main.py --gan_type gan --net_type cnn --epoch 10 --clip 0 --D_iter 1
python main.py --gan_type gan --net_type mlp --epoch 20 --clip 0 --D_iter 5
echo "train wgan with gradient clip"
python main.py --gan_type wgan --net_type cnn --epoch 10 --clip 0.05 --D_iter 5
python main.py --gan_type wgan --net_type mlp --epoch 20 --clip 0.1 --D_iter 5
echo "train improved wgan with gradient penalty"
python main.py --gan_type wgan_gp --net_type cnn --epoch 10 --clip 0 --D_iter 1
python main.py --gan_type wgan_gp --net_type mlp --epoch 20 --clip 0 --D_iter 1
echo "train lsgan"
python main.py --gan_type lsgan --net_type cnn --epoch 10 --clip 0.05 --D_iter 1
python main.py --gan_type lsgan --net_type mlp --epoch 20 --clip 0.05 --D_iter 1
echo "train dragan"
python main.py --gan_type dragan --net_type cnn --epoch 10 --clip 0 --D_iter 1
python main.py --gan_type dragan --net_type mlp --epoch 20 --clip 0 --D_iter 5
echo "train cgan"
python main.py --gan_type cgan --net_type cnn --epoch 10 --D_iter 1
python main.py --gan_type cgan --net_type mlp --epoch 20 --D_iter 1
echo "train infogan"
python main.py --gan_type infogan --net_type cnn --epoch 10 --D_iter 1
python main.py --gan_type infogan --net_type mlp --epoch 20 --D_iter 5