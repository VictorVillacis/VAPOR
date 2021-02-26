from tensorflow import keras
import tensorflow as tf

import os
import sys
sys.path.append(os.path.realpath('.'))

from data.encode_decode import char_dict


class VAPORASR(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self):

        super(VAPORASR, self).__init__()

        self.conv = keras.layers.Conv2D(filters=64, kernel_size=5, padding='same')
        self.lstm = keras.layers.LSTM(units=64, return_sequences=True)
        self.dense = keras.layers.Dense(units=29, activation='sigmoid')

        return

    # @tf.Module.with_name_scope
    def call(self, inputs, x_len=None, training=False, mask=None):
        x = self.conv(inputs)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [x.shape[0], -1, x.shape[2] * x.shape[3]])
        x = self.lstm(x)
        x = self.dense(x)

        if not training:
            x = tf.transpose(x, perm=[1, 0, 2])
            ctc_decoding = tf.nn.ctc_beam_search_decoder(inputs=x, sequence_length=x_len)
            max_prediction_index = tf.math.argmax(ctc_decoding[1]).numpy()[0]
            dense_predictions = tf.sparse.to_dense(ctc_decoding[0][0], default_value=char_dict['_'])
            max_prediction = dense_predictions[max_prediction_index]
            return max_prediction
        else:
            return x


