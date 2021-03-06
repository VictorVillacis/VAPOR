
import tensorflow as tf
from data.data_loader import SpeechDataset
from model.model import VAPORASR
from data.encode_decode import int_sequence_to_str
from nltk.metrics.distance import edit_distance
from tqdm import tqdm


def test_step(model, x, x_len, y, y_len):

    out = model(x, x_len, training=False)

    word_error_rate = calculate_word_error_rate(out, y, y_len)

    return word_error_rate


def calculate_word_error_rate(out, y, y_len):

    batch_size = len(out)

    word_error_rate = 0

    for batch_index in range(batch_size):
        batch_pred_str = int_sequence_to_str(out[batch_index])
        batch_y = y[batch_index]
        batch_y_len = y_len[batch_index]
        batch_y = batch_y[:batch_y_len]
        batch_y_str = int_sequence_to_str(batch_y)

        word_error_rate += edit_distance(batch_pred_str, batch_y_str)

    word_error_rate /= batch_size

    return word_error_rate


def test(model, opt):

    test_dataset = SpeechDataset(opt['test_path'], opt['batch_size'])
    num_batches = len(test_dataset)
    test_dataset = tf.data.Dataset.from_generator(test_dataset,
                                                  output_signature=test_dataset.get_output_tensor_signatures())

    word_error_rate = 0

    for step, batch in tqdm(enumerate(test_dataset), total=num_batches):
        x, x_len, y, y_len = batch
        word_error_rate += test_step(model, x, x_len, y, y_len)

    word_error_rate /= num_batches

    print("Epoch Word Error Rate: {:.3f}".format(word_error_rate))

    return


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