from data.data_loader import SpeechDataset
import tensorflow as tf


def train(opt: dict):

    train_dataset = SpeechDataset(opt['train_path'], options['batch_size'])
    train_dataset = tf.data.Dataset.from_generator(train_dataset,
                                                   output_signature=train_dataset.get_output_tensor_signatures())

    tmp4 = list(train_dataset.take(3))

    return


if __name__ == "__main__":

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"
    options['batch_size'] = 8

    train(options)

    pass



