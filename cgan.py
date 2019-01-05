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

class CGAN(BasicTrainFramework):
    def __init__(self,
                 optim_type='adam',
                 net_type='cnn',
                 batch_size=64,
                 noise_dim=50,
                 learning_rate=2e-4,
                 optim_num=0.5,
                 critic_iter=1,
                 plot_iter=5,
                 verbose=True):
        
        self.noise_dim = noise_dim
        self.class_num = 10
        self.lr = learning_rate
        self.optim_num = optim_num
        self.critic_iter = critic_iter
        self.plot_iter = plot_iter
        self.verbose = verbose
        super(CGAN, self).__init__(batch_size, "cgan_"+net_type)

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
        self.G = self.generator(self.noise, self.labels, is_training=True, reuse=False)
        self.G_test = self.generator(self.noise, self.labels, is_training=False, reuse=True)
        self.logit_real, self.conv_real, self.cls_real = self.discriminator(self.source, self.labels, is_training=True, reuse=False)
        self.logit_fake, self.conv_fake, self.cls_fake = self.discriminator(self.G, self.labels, is_training=True, reuse=True)
    
    def build_optimizer(self, optim_type='adam'):
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))

        if optim_type == 'adam':
            self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.G_loss, var_list=self.generator.vars)
        elif optim_type == 'rmsprop':
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        
    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum])
    
    def sample(self, epoch):
        print "sample at {}".format(epoch)
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
        
        N = 5
        for i in range(N):
            for j in range(N):
                idx = i*N + j
                plt.subplot(N, N, idx+1)
                plt.imshow(G[idx, :, :, 0], cmap=plt.cm.gray)
                plt.title(convert_label(np.argmax(data["labels"][idx])))
                plt.xticks([])
                plt.yticks([])
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.savefig(os.path.join(self.fig_dir, "fake_{}.png".format(epoch)))
        plt.clf()

        if epoch == 0:
            for i in range(N):
                for j in range(N):
                    idx = i*N + j
                    plt.subplot(N, N, idx+1)
                    plt.imshow(data["data"][idx, :, :, 0], cmap=plt.cm.gray)
                    plt.title(convert_label(np.argmax(data["labels"][idx])))
                    plt.xticks([])
                    plt.yticks([])
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
            plt.savefig(os.path.join(self.fig_dir, "real_{}.png".format(epoch)))
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

                # train D
                self.sess.run(self.D_solver, feed_dict=feed_dict)

                # train G
                if (cnt-1) % self.critic_iter == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                    
                if cnt % 10 == 0:
                    d_loss, d_loss_r, d_loss_f, g_loss, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.summary], feed_dict=feed_dict)
                    if self.verbose:
                        print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f" % \
                            (epoch, epoches, idx, batches_per_epoch, d_loss, d_loss_r, d_loss_f, g_loss)
                    self.writer.add_summary(sum_str, cnt)

            if epoch % self.plot_iter == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)
                
        