# coding:utf-8

import tensorflow as tf 
from utils import BasicBlock, conv2d, bn, lrelu, dense, conv_cond_concat

class Classifier_MLP(BasicBlock):
    def __init__(self, class_num, name=None):
        super(Classifier_MLP, self).__init__(None, name or 'Classifier_CNN')
        self.class_num = class_num
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse):
            net = lrelu(bn(dense(x, 64, name='c_fc1'), is_training, name='c_bn1'), name='c_l1')
            out_logit = dense(net, self.class_num, name='c_l2')
            out = tf.nn.softmax(out_logit)

            return out_logit, out