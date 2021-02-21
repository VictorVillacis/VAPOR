
import tensorflow as tf
from data.data_loader import SpeechDataset
from model.model import VAPORASR


def test_step(model, x, x_len, y, y_len):

    out = model(x, x_len, training=False)

    return


def test(model, opt):

    test_dataset = SpeechDataset(opt['test_path'], opt['batch_size'])
    test_dataset = tf.data.Dataset.from_generator(test_dataset,
                                                  output_signature=test_dataset.get_output_tensor_signatures())

    for step, batch in enumerate(test_dataset):
        x, x_len, y, y_len = batch
        test_step(model, x, x_len, y, y_len)


if __name__ == '__main__':

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"
    options['batch_size'] = 8
    options['Epochs'] = 8
    options['model'] = "model/saves/exp7/"
    model = VAPORASR()
    model.load_weights(options['model'])
    test(model, options)

    pass