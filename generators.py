# coding:utf-8

import tensorflow as tf 
from utils import BasicBlock, dense, deconv2d, bn

class Generator_CNN(BasicBlock):
    def __init__(self, name=None):
        super(Generator_CNN, self).__init__(None, name or "Generator_CNN")
    
    def __call__(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = z.get_shape().as_list()[0]
            if y is not None:
                z = tf.concat([z,y], 1) # [bz,zdim+10]

            net = tf.nn.relu(bn(dense(z, 1024, name='g_fc1'), is_training, name='g_bn1'))
            net = tf.nn.relu(bn(dense(net, 128*7*7, name='g_fc2'), is_training, name='g_bn2'))
            net = tf.reshape(net, [batch_size, 7, 7, 128])
            # [bz, 14, 14, 64]
            net = tf.nn.relu(bn(deconv2d(net, 64, 4, 4, 2, 2, padding='SAME', name='g_dc3'), is_training, name='g_bn3'))
            # [bz, 28, 28, 1]
            out = tf.nn.sigmoid(deconv2d(net, 1, 4, 4, 2, 2, padding='SAME', name='g_dc4'))
            return out
            
class Generator_MLP(BasicBlock):
    def __init__(self, name=None):
        super(Generator_MLP, self).__init__(None, name or "Generator_MLP")
    
    def __call__(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = z.get_shape().as_list()[0]
            if y is not None:
                z = tf.concat([z,y], 1)
            
            net = tf.nn.relu(bn(dense(z, 128, name='g_fc1'), is_training, name='g_bn1'))
            net = tf.nn.relu(bn(dense(net, 256, name='g_fc2'), is_training, name='g_bn2'))
            net = tf.nn.relu(bn(dense(net, 512, name='g_fc3'), is_training, name='g_bn3'))
            net = tf.nn.relu(bn(dense(net, 1024, name='g_fc4'), is_training, name='g_bn4'))
            net = tf.nn.sigmoid(dense(net, 784, name='g_fc5'))

            out = tf.reshape(net, (batch_size, 28, 28, 1))
            return out


    
