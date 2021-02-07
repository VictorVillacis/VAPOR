from pathlib import Path
from tqdm import tqdm
import logging
import sys
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from encode_decode import char_str_to_number_seq, char_dict
import os
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.ioff()

random.seed(0)
np.random.seed(0)


class SpeechDataset(object):

    def __init__(self, data_dir_: str, batch_size: int = 8, verbose_level: int = 0):

        # Initialize class attributes
        self.data_dir = Path(data_dir_)
        self.data = []
        self.data_index = 0
        self.vis = Visualizer()
        self.batch_size = batch_size
        self.verbose_level = verbose_level

        logger.info("Loading Data...")
        self.load_data_paths()

        # Get number of batches, cut data to whole batches only
        self.batches = len(self.data) // self.batch_size
        self.data = self.data[: self.batches * self.batch_size]

        # Initial data shuffle
        self.shuffle()
        return

    def load_data_paths(self):
        # Data uses Librespeech dataset: http://www.openslr.org/12
        # Data is organized as voice_dir > recording_dir > sample_path, with the label .txt file being in the same
        # directory as the samples

        # For every voice...
        num_voices = len(list(self.data_dir.iterdir()))
        for voice_dir in tqdm(self.data_dir.iterdir(), total=num_voices):
            if voice_dir.is_dir():

                # For every recording...
                for recording_dir in voice_dir.iterdir():

                    sample_label = None
                    current_samples = {}

                    # Divide samples into .flac audio files and .txt label file
                    for sample_path in recording_dir.iterdir():
                        if sample_path.suffix == '.txt':
                            sample_label = sample_path
                        elif sample_path.suffix == ".flac":
                            current_samples[sample_path.stem] = sample_path

                    # Open label file
                    with open(str(sample_label), 'r') as current_labels:

                        # For every label in label file...
                        for line in current_labels.readlines():

                            # Label file is organized as lines of "AudioSample Label"
                            # Get the name of the audio sample
                            line_sample = line.split(" ")[0]

                            # If the audio sample name was previously found...
                            if line_sample in current_samples:
                                # Get the file name, label string
                                sample_file = current_samples[line_sample]
                                line_label = line[len(line_sample) + 1:]

                                # Append to self.data
                                self.data.append({'x': sample_file, 'y': line_label})

        return

    def shuffle(self):
        random.shuffle(self.data)
        return

    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def __call__(self, *args, **kwargs):
        while True:
            yield self.next_batch()

    def next_batch(self):

        # Initialize batch objects
        x_, x_lengths, y_, y_lengths = [None] * self.batch_size, \
                                       [None] * self.batch_size, \
                                       [None] * self.batch_size, \
                                       [None] * self.batch_size

        # Get batch range
        current_data_indices = list(range(self.data_index, self.data_index + self.batch_size))
        self.data_index += self.batch_size

        # Accumulate batch data
        for batch_index, current_data_index in enumerate(current_data_indices):
            next_x, next_x_length, next_y, next_y_length = self.get_current_raw_data(self.data, current_data_index)

            x_[batch_index] = next_x
            x_lengths[batch_index] = next_x_length
            y_[batch_index] = next_y
            y_lengths[batch_index] = next_y_length

        # Collate data; formats N X to NHW, N Y to NL
        x_, x_lengths, y_, y_lengths = self.collate(x_, x_lengths, y_, y_lengths)

        return x_, x_lengths, y_, y_lengths

    def collate(self, x_, x_lengths, y_, y_lengths):

        # Find maximum X, Y length
        max_x_len = max(x_lengths)
        max_y_len = max(y_lengths)

        # Pad all X to maximum label length
        for iter_1 in range(self.batch_size):
            x_length_difference = max_x_len - x_lengths[iter_1]
            x_pad = [[0, 0], [0, x_length_difference]]
            x_[iter_1] = np.pad(x_[iter_1], x_pad, constant_values=char_dict['_'])

        # Pad all Y to maximum label length
        for iter_1 in range(self.batch_size):
            y_length_difference = max_y_len - y_lengths[iter_1]
            y_pad = [0, y_length_difference]
            y_[iter_1] = np.pad(y_[iter_1], y_pad, constant_values=char_dict['_'])

        # # Stack X, X lengths, Y, Y lengths into batch
        x_ = np.stack(x_)
        x_lengths = np.stack(x_lengths)
        y_ = np.stack(y_)
        y_lengths = np.stack(y_lengths)

        return x_, x_lengths, y_, y_lengths

    def next_item(self):
        x_, x_length, y_, y_length = self.get_current_raw_data(self.data, self.data_index)
        self.increment_index()
        return x_, x_length, y_, y_length

    def increment_index(self):
        self.data_index += 1
        if self.data_index > len(self):
            self.shuffle()
            self.reset_index()
        return

    def reset_index(self):
        self.data_index = 0
        return

    def __getitem__(self, item):
        return self.get_current_raw_data(self.data, item)

    @staticmethod
    def get_current_raw_data(data, index):
        # Get current data pair
        current_data = data[index]
        x_path, y_ = current_data['x'], current_data['y']

        # Read in X
        x_data, sample_rate = librosa.load(x_path)

        # Process X into mel spectrogram
        x_ = librosa.feature.melspectrogram(x_data, sample_rate)
        x_ = np.log(x_ + 1e-14)

        # Process Y into number sequence
        y_ = char_str_to_number_seq(y_)

        # Get lengths of X and Y
        x_length = x_.shape[1]
        y_length = y_.shape[0]

        return x_, x_length, y_, y_length

    def visualize(self, x_data, y_):
        # Use visualization class to show X, Y if verbose level is high enough
        if self.verbose_level > 1:
            self.vis.visualize_audio(x_data, y_)
        return

    def get_output_tensor_signatures(self):
        # Get sample of data output
        x, x_lens, y, y_lens = self.next_batch()
        # De-increment data index to prior value
        self.data_index -= 1

        # Create tensor specifications from sample
        x_spec = tf.TensorSpec(shape=(x.shape[0], x.shape[1], None), dtype=x.dtype)
        x_len_spec = tf.TensorSpec(shape=x_lens.shape, dtype=x_lens.dtype)
        y_spec = tf.TensorSpec(shape=(y.shape[0], None), dtype=y.dtype)
        y_len_spec = tf.TensorSpec(shape=y_lens.shape, dtype=y_lens.dtype)

        # Return specifications
        return x_spec, x_len_spec, y_spec, y_len_spec


class Visualizer:

    def __init__(self):
        return

    @staticmethod
    def visualize_audio(x_data, y_):

        # Create matplotlib figure, axis
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Visualize data
        ax.imshow(x_data)

        # Set axis labels and title
        ax.set_title("\n".join(wrap(y_, 55)), pad=20)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        # Adjust for long titles, show figure
        fig.subplots_adjust(top=0.8)
        plt.show()


if __name__ == "__main__":

    data_dir = "raw/Dev/"
    dataset = SpeechDataset(data_dir, verbose_level=1)

    tmp = dataset.next_batch()
    pass


