from tensorflow import keras
import tensorflow as tf




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
            sparse_predictions = ctc_decoding[0][0]

            batch_size = x.shape[1]
            batch_predictions = [None] * batch_size

            for batch_prediction in range(batch_size):

                batch_item_indices = sparse_predictions.indices[:, 0] == batch_prediction
                batch_item_values = sparse_predictions.values[batch_item_indices]
                batch_predictions[batch_prediction] = batch_item_values

            return batch_predictions
        else:
            return x


