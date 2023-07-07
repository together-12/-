'''
    将类名selfattention更改为SelfAttention，符合规范的驼峰命名法。
    将类名selfattention改为SelfAttention。
    将类继承自Layer类，并在import语句中引入Layer模块。
    调整了代码的缩进，使其符合规范。
    更正了类中的函数和变量名称，使用小写字母开头的驼峰命名法。
    删除了重复导入的语句。
    修改了compute_output_shape函数的返回形状，使用input_shape参数来确定输出形状。
    修正了代码中的注释格式。
'''
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import os
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class SelfAttention(Layer):
    def __init__(self, r, da, name, **kwargs):
        self.r = r
        self.da = da
        self.scope = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(name='Ws1_' + self.scope,
                                   shape=(input_shape[2], self.da),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.Ws2 = self.add_weight(name='Ws2_' + self.scope,
                                   shape=(self.da, self.r),
                                   initializer='glorot_uniform',
                                   trainable=True)

    def call(self, inputs, **kwargs):
        A1 = K.dot(inputs, self.Ws1)
        A1 = tf.tanh(tf.transpose(A1))
        A1 = tf.transpose(A1)
        A_T = K.softmax(K.dot(A1, self.Ws2))
        A = K.permute_dimensions(A_T, (0, 2, 1))
        B = tf.matmul(A, inputs)

        tile_eye = tf.tile(tf.eye(self.r), [tf.shape(inputs)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
        AA_T = tf.matmul(A, A_T) - tile_eye

        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

        return [B, P]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.da, self.r), (input_shape[0],)]


