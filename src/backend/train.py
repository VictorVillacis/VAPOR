# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data.data_loader import SpeechDataset
import tensorflow as tf
from model.model import VAPORASR
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from os import mkdir
from tqdm import tqdm

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


def get_current_save_dir(save_dir):
    save_dir = Path(save_dir)
    sub_dirs = [int(x.name[3:]) for x in save_dir.iterdir() if x.is_dir()]

    if len(sub_dirs) == 0:
        mkdir(str(save_dir) + '/exp0/')
        return str(save_dir) + '/exp0/'
    else:
        max_experiment = max(sub_dirs)
        mkdir(str(save_dir) + '/exp{}/'.format(max_experiment + 1))

    return str(save_dir) + '/exp{}/'.format(max_experiment + 1)


def save_model(model, opt):
    model.save_weights(opt['current_save_dir'])
    return


def train(opt: dict):

    train_dataset = SpeechDataset(opt['train_path'], options['batch_size'])
    num_batches = len(train_dataset)
    train_dataset = tf.data.Dataset.from_generator(train_dataset,
                                                   output_signature=train_dataset.get_output_tensor_signatures())

    model = VAPORASR()
    optimizer = Adam()

    save_model(model, opt)

    for epoch in range(opt['epochs']):

        train_dataset = train_dataset.shuffle(buffer_size=64)

        for step, batch in tqdm(enumerate(train_dataset), total=num_batches):
            x, x_len, y, y_len = batch
            train_step(model, optimizer, x, x_len, y, y_len)

        save_model(model, opt)

    return


if __name__ == "__main__":

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"
    options['batch_size'] = 8
    options['epochs'] = 8
    options['save_dir'] = 'model/saves/'
    options['current_save_dir'] = get_current_save_dir(options['save_dir'])
    train(options)

    pass



