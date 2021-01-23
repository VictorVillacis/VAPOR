from data.data_loader import SpeechDataset
import tensorflow as tf


def train(opt: dict):

    train_dataset = SpeechDataset(opt['train_path'])
    train_dataset = tf.data.Dataset.from_generator(
        train_dataset,
        output_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, None), dtype=tf.float32)))



    return


if __name__ == "__main__":

    options = dict()
    options['train_path'] = "data/raw/Dev/"
    options['test_path'] = "data/raw/Dev/"

    train(options)

    pass



