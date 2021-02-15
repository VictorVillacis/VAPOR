# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data.data_loader import SpeechDataset
import tensorflow as tf
from model.model import VAPORASR
from tensorflow.keras.optimizers import Adam

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, x, x_len, y, y_len):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss = tf.nn.ctc_loss(y, out, x_len, y_len, logits_time_major=False)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(opt: dict):

    train_dataset = SpeechDataset(opt['train_path'], options['batch_size'])
    num_batches = len(train_dataset)
    train_dataset = tf.data.Dataset.from_generator(train_dataset,
                                                   output_signature=train_dataset.get_output_tensor_signatures())

    model = VAPORASR()
    optimizer = Adam()

    for epoch in range(opt['Epochs']):

        train_dataset = train_dataset.shuffle(buffer_size=64)

        for step, batch in enumerate(train_dataset):
            x, x_len, y, y_len = batch
            train_step(model, optimizer, x, x_len, y, y_len)

            print(step)

    return


if __name__ == "__main__":

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"
    options['batch_size'] = 8
    options['Epochs'] = 8
    train(options)

    pass



