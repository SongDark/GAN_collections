# coding:utf-8

import matplotlib, os
matplotlib.use("Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from utils import BasicTrainFramework
from datamanager import datamanager_mnist
from discriminators import Discriminator_CNN, Discriminator_MLP
from generators import Generator_CNN, Generator_MLP
from matplotlib import pyplot as plt

class GAN(BasicTrainFramework):
    def __init__(self, 
                gan_type='gan',
                net_type='cnn',
                optim_type='adam',
                batch_size=64, 
                noise_dim=50,
                learning_rate=2e-4,
                optim_num=0.5,
                clip_num=0.03,
                critic_iter=5,
                plot_iter=5,
                verbose=True
                ):
        
        self.noise_dim = noise_dim
        self.class_num = 10
        self.clip_num = None if clip_num==0 else clip_num
        self.lr = learning_rate
        self.optim_num = optim_num
        self.critic_iter = critic_iter
        self.plot_iter = plot_iter
        self.verbose = verbose
        super(GAN, self).__init__(batch_size, gan_type+"_"+net_type)
        
        self.gan_type = gan_type
        self.optim_type = optim_type

        self.data = datamanager_mnist(train_ratio=1.0, fold_k=None, norm=True, expand_dim=True, seed=23333)
        sample_data = datamanager_mnist(train_ratio=1.0, fold_k=None, norm=True, expand_dim=True, seed=23333)
        self.sample_data = sample_data(self.batch_size, var_list=["data", "labels"])

        if net_type == 'cnn':
            self.generator = Generator_CNN('cnn_generator')
            self.discriminator = Discriminator_CNN(class_num=self.class_num, name='cnn_discriminator')
        elif net_type == 'mlp':
            self.generator = Generator_MLP('mlp_generator')
            self.discriminator = Discriminator_MLP(class_num=self.class_num, name='mlp_discriminator')

        self.build_placeholder()
        self.build_gan()
        self.build_optimizer(optim_type)
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.noise = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
        self.source = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)

    def build_gan(self):
        self.G = self.generator(self.noise, is_training=True, reuse=False)
        self.G_test = self.generator(self.noise, is_training=False, reuse=True)
        self.logit_real, self.conv_real, self.cls_real = self.discriminator(self.source, is_training=True, reuse=False)
        self.logit_fake, self.conv_fake, self.cls_fake = self.discriminator(self.G, is_training=True, reuse=True)

        self.cls_real_softmax = tf.nn.softmax(self.cls_real)
        self.cls_fake_softmax = tf.nn.softmax(self.cls_fake)
        self.batch_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.cls_real_softmax, axis=1), tf.argmax(self.labels, axis=1))))

    def build_optimizer(self, optim_type='adam'):
        if self.gan_type == 'gan':
            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        elif self.gan_type == 'wgan':
            self.D_loss_real = - tf.reduce_mean(self.logit_real) 
            self.D_loss_fake = tf.reduce_mean(self.logit_fake)
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = - self.D_loss_fake
            if self.clip_num:
                print "GC"
                self.D_clip = [v.assign(tf.clip_by_value(v, -self.clip_num, self.clip_num)) for v in self.discriminator.vars]
        elif self.gan_type == 'wgan_gp':
            self.D_loss_real = - tf.reduce_mean(self.logit_real)  
            self.D_loss_fake = tf.reduce_mean(self.logit_fake)
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = - self.D_loss_fake
            # Gradient Penalty
            alpha = tf.random_uniform(shape=self.source.get_shape(), minval=0.0, maxval=1.0)
            differences = self.G - self.source
            interpolates = self.source + alpha * differences
            yd, _, _ = self.discriminator(interpolates, is_training=True, reuse=True)
            gradients = tf.gradients(yd, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0)**2)
            self.D_loss += 0.25 * gradient_penalty
        elif self.gan_type == 'dragan':
            self.source_p = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
            # DRAGAN Gradient Penalty
            alpha = tf.random_uniform(shape=self.source.get_shape(), minval=0.0, maxval=1.0)
            differences = self.source_p - self.source # difference from WGAP-GP
            interpolates = self.source + alpha * differences
            yd, _, _ = self.discriminator(interpolates, is_training=True, reuse=True)
            gradients = tf.gradients(yd, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0)**2)
            self.D_loss += 0.25 * gradient_penalty
        elif self.gan_type == 'lsgan':
            def mse_loss(pred, data):
                return tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.batch_size
            self.D_loss_real = tf.reduce_mean(mse_loss(self.logit_real, tf.ones_like(self.logit_real)))
            self.D_loss_fake = tf.reduce_mean(mse_loss(self.logit_fake, tf.zeros_like(self.logit_fake)))
            self.D_loss = 0.5 * (self.D_loss_real + self.D_loss_fake)
            self.G_loss = tf.reduce_mean(mse_loss(self.logit_fake, tf.ones_like(self.logit_fake)))
            if self.clip_num:
                print "GC"
                self.D_clip = [v.assign(tf.clip_by_value(v, -self.clip_num, self.clip_num)) for v in self.discriminator.vars]


        self.C_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.cls_real))

        if optim_type == 'adam':
            self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.G_loss, var_list=self.generator.vars)
        elif optim_type == 'rmsprop':
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        
        self.C_solver_real = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.C_loss_real, var_list=self.discriminator.vars)

    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        C_sum_real = tf.summary.scalar("C_loss_real", self.C_loss_real)
        ACC_sum_real = tf.summary.scalar("Batch_Acc_real", self.batch_acc)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum, C_sum_real, ACC_sum_real])
        
    
    def sample(self, epoch):
        def convert_label(i):
            return chr(i + 48)
        data = self.sample_data
        feed_dict = {
            self.source: data["data"],
            self.labels: data["labels"],
            # self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
            self.noise: np.random.normal(size=(self.batch_size, self.noise_dim))
        }
        G = self.sess.run(self.G_test, feed_dict=feed_dict)
        cls_r, cls_f = self.sess.run([self.cls_real_softmax, self.cls_fake_softmax], feed_dict=feed_dict)

        N = 5
        for i in range(N):
            for j in range(N):
                idx = i*N + j
                plt.subplot(N, N, idx+1)
                plt.imshow(G[idx, :, :, 0], cmap=plt.cm.gray)
                tmp = [np.argmax(data["labels"][idx]), np.argmax(cls_r[idx]), np.argmax(cls_f[idx])]
                plt.title("_".join([convert_label(s) for s in tmp]))
                plt.xticks([])
                plt.yticks([])
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.savefig(os.path.join(self.fig_dir, "fake_{}.png".format(epoch)))
        plt.clf()

    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data", "labels"])
                
                feed_dict = {
                    self.source: data["data"],
                    self.labels: data["labels"],
                    # self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
                    self.noise: np.random.normal(size=(self.batch_size, self.noise_dim))
                }
                if self.gan_type == 'dragan':
                    feed_dict[self.source_p] = data["data"] + 0.5*data["data"].std()*np.random.random(data["data"].shape)

                # train D
                self.sess.run([self.D_solver, self.C_solver_real], feed_dict=feed_dict)
                if self.clip_num:
                    self.sess.run(self.D_clip)

                # train G
                if (cnt-1) % self.critic_iter == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                    
                if cnt % 10 == 0:
                    d_loss, d_loss_r, d_loss_f, g_loss, c_loss_r, acc, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.C_loss_real, self.batch_acc, self.summary], feed_dict=feed_dict)
                    if self.verbose:
                        print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f Cr=%.3f Acc=%.2f" % \
                            (epoch, epoches, idx, batches_per_epoch, d_loss, d_loss_r, d_loss_f, g_loss, c_loss_r, acc)
                    self.writer.add_summary(sum_str, cnt)

            if epoch % self.plot_iter == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)
               