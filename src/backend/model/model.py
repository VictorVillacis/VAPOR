from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf


class VAPORASR(keras.Model):

    def get_config(self):
        pass

    def __init__(self):

        super(VAPORASR, self).__init__()

        self.conv = keras.layers.Conv2D(filters=64, kernel_size=5)
        self.lstm = keras.layers.LSTM(units=64, return_sequences=True)
        self.dense = keras.layers.Dense(units=29, activation='sigmoid')
        return

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.lstm(x)
        x = self.dense(x)

        return x


