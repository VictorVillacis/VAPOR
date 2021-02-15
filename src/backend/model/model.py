from tensorflow import keras
import tensorflow as tf


class VAPORASR(tf.Module):

    def get_config(self):
        pass

    def __init__(self):

        super(VAPORASR, self).__init__()

        self.conv = keras.layers.Conv2D(filters=64, kernel_size=5)
        self.lstm = keras.layers.LSTM(units=64, return_sequences=True)
        self.dense = keras.layers.Dense(units=29, activation='sigmoid')

        return

    @tf.Module.with_name_scope
    def __call__(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [x.shape[0], -1, x.shape[2] * x.shape[3]])
        x = self.lstm(x)
        x = self.dense(x)

        return x


