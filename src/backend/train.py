from data.data_loader import SpeechDataset
import tensorflow as tf
from tensorflow.keras import backend as K
from model.model import VAPORASR

tf_sess = tf.Session()
K.set_session(tf_sess)


def train(opt: dict):

    train_dataset = SpeechDataset(opt['train_path'], options['batch_size'])
    train_dataset = tf.data.Dataset.from_generator(train_dataset,
                                                   output_signature=train_dataset.get_output_tensor_signatures())

    model = VAPORASR()
    tmp = list(train_dataset.take(1))
    x, x_len, y, y_len = tmp[0]
    out = model(x)

    loss = tf.nn.ctc_loss(y, out, x_len, y_len, logits_time_major=False)

    return


if __name__ == "__main__":

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"
    options['batch_size'] = 8

    train(options)

    pass



