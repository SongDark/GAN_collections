# coding:utf-8

import matplotlib, os
matplotlib.use("Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from utils import BasicTrainFramework, one_hot_encode
from datamanager import datamanager_mnist
from discriminators import Discriminator_CNN, Discriminator_MLP
from generators import Generator_CNN, Generator_MLP
from classifier import Classifier_MLP
from matplotlib import pyplot as plt

class InfoGAN(BasicTrainFramework):
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
        super(InfoGAN, self).__init__(batch_size, "infogan_"+net_type)

        # code
        self.len_discrete_code = self.class_num  # categorical distribution (i.e. label)
        self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)

        self.optim_type = optim_type
        self.SUPERVISED = True

        self.data = datamanager_mnist(train_ratio=1.0, fold_k=None, norm=True, expand_dim=True, seed=23333)
        sample_data = datamanager_mnist(train_ratio=1.0, fold_k=None, norm=True, expand_dim=True, seed=23333)
        self.sample_data = sample_data(self.batch_size, var_list=["data", "labels"])

        if net_type == 'cnn':
            self.generator = Generator_CNN(name='cnn_generator')
            self.discriminator = Discriminator_CNN(name='cnn_discriminator')
        elif net_type == 'mlp':
            self.generator = Generator_MLP(name='mlp_generator')
            self.discriminator = Discriminator_CNN(name='discriminator')
        self.classifier = Classifier_MLP(self.len_discrete_code + self.len_continuous_code, name='classifier')

        self.build_placeholder()
        self.build_gan()
        self.build_optimizer(optim_type)
        self.build_summary()

        self.build_sess()
        self.build_dirs()

    def build_placeholder(self):
        self.noise = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
        self.source = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.len_discrete_code + self.len_continuous_code), dtype=tf.float32)

    def build_gan(self):
        self.G = self.generator(self.noise, self.labels, is_training=True, reuse=False)
        self.G_test = self.generator(self.noise, self.labels, is_training=False, reuse=True)
        self.logit_real, self.conv_real = self.discriminator(self.source, is_training=True, reuse=False)
        self.logit_fake, self.conv_fake = self.discriminator(self.G, is_training=True, reuse=True)

        self.logit_cls, self.softmax_cls = self.classifier(self.conv_fake, is_training=True)

    def build_optimizer(self, optim_type='adam'):
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        
        # discrete code : categorical
        disc_code_est = self.softmax_cls[:, :self.len_discrete_code]
        disc_code_tg = self.labels[:, :self.len_discrete_code]
        q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg))
        # continuous code : gaussian
        cont_code_est = self.softmax_cls[:, self.len_discrete_code:]
        cont_code_tg = self.labels[:, self.len_discrete_code:]
        q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))
        self.Q_loss = q_disc_loss + q_cont_loss

        if optim_type == 'adam':
            self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.optim_num).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=5*self.lr, beta1=self.optim_num).minimize(self.G_loss, var_list=self.generator.vars)
        elif optim_type == 'rmsprop':
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=5*self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        
        self.Q_solver = tf.train.AdamOptimizer(learning_rate=5*self.lr, beta1=self.optim_num).minimize(self.Q_loss, var_list=self.classifier.vars + self.generator.vars + self.discriminator.vars)

    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        Q_sum = tf.summary.scalar("Q_loss", self.Q_loss)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum, Q_sum])
    
    def sample(self, epoch):
        # batch_size=64
        labels = np.repeat(np.arange(10).reshape((10,1)), 10)
        labels = one_hot_encode(labels, 10) # [100, 10]
        test_codes = np.concatenate([labels, np.tile(np.linspace(-1, 1, 10), 10)[:, None], np.zeros((100, 1))], 1)
        test_codes = np.concatenate([test_codes, test_codes], 0)

        feed_dict = {
            self.labels: test_codes[:self.batch_size, :],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        G1 = self.sess.run(self.G_test, feed_dict=feed_dict)
        feed_dict = {
            self.labels: test_codes[self.batch_size:2*self.batch_size, :],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        G1 = np.concatenate([G1, self.sess.run(self.G_test, feed_dict=feed_dict)], 0)
        G1 = np.concatenate([np.concatenate([seq[:,:,0] for seq in G1[10*i:10*(i+1)]], 1) for i in range(10)], 0)
        plt.imshow(G1, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.fig_dir, "fake_{}_1.png".format(epoch)))
        plt.clf()

        test_codes = np.concatenate([labels, np.zeros((100, 1)), np.tile(np.linspace(-1, 1, 10), 10)[:, None]], 1)
        test_codes = np.concatenate([test_codes, test_codes], 0)

        feed_dict = {
            self.labels: test_codes[:self.batch_size, :],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        G2 = self.sess.run(self.G_test, feed_dict=feed_dict)
        feed_dict = {
            self.labels: test_codes[self.batch_size:2*self.batch_size, :],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        G2 = np.concatenate([G2, self.sess.run(self.G_test, feed_dict=feed_dict)], 0)
        G2 = np.concatenate([np.concatenate([seq[:,:,0] for seq in G2[10*i:10*(i+1)]], 1) for i in range(10)], 0)
        plt.imshow(G2, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.fig_dir, "fake_{}_2.png".format(epoch)))
        plt.clf()

    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data", "labels"])
                
                # [bz, 10]
                if self.SUPERVISED:
                    batch_labels = data["labels"] 
                else:
                    batch_labels = np.random.multinomial(1, self.len_discrete_code * [float(1.0/self.len_discrete_code)], size=[self.batch_size])
                # [bz, 12]
                batch_codes = np.concatenate((batch_labels, np.random.uniform(-1,1,size=(self.batch_size, 2))), axis=1)

                feed_dict = {
                    self.source: data["data"],
                    self.labels: batch_codes,
                    self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
                }

                # train D
                self.sess.run(self.D_solver, feed_dict=feed_dict)

                # train G
                if (cnt-1) % self.critic_iter == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                
                # train Q 
                self.sess.run(self.Q_solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    d_loss, d_loss_r, d_loss_f, g_loss, q_loss, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.Q_loss, self.summary], feed_dict=feed_dict)
                    if self.verbose:
                        print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f Q=%.3f" % \
                            (epoch, epoches, idx, batches_per_epoch, d_loss, d_loss_r, d_loss_f, g_loss, q_loss)
                    self.writer.add_summary(sum_str, cnt)

            if epoch % self.plot_iter == 0:
                self.sample(epoch)
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)
               