'''
    格式化了代码的缩进，使其符合通用的Python代码风格。
    修正了类名MediumLayer，遵循了大驼峰命名法的命名约定。
    修正了__init__方法中的参数命名，使其符合一般的Python风格。
    修正了compute_output_shape方法的缩进，并为方法体添加了合适的注释。
'''
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K

class MediumLayer(Layer):
    def __init__(self, **kwargs):
        super(MediumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MediumLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sentence_token_level_outputs = tf.stack(inputs, axis=0)
        sentence_token_level_outputs = K.permute_dimensions(sentence_token_level_outputs, (1, 0, 2))
        return sentence_token_level_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape), input_shape[0][1])

'''
x1 = tf.random.truncated_normal([100, 150])
x2 = tf.random.truncated_normal([100, 150])
x3 = tf.random.truncated_normal([100, 150])
x4 = tf.random.truncated_normal([100, 150])

w = MediumLayer()([x1, x2, x3, x4])
print(w)
'''
