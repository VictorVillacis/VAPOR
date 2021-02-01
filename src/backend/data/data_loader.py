
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import soundfile
import random
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from encode_decode import char_str_to_number_seq, char_dict
import tensorflow as tf
import tensorflow_io as tfio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

plt.ioff()

random.seed(0)
np.random.seed(0)


class SpeechDataset(object):

    def __init__(self, data_dir_: str, batch_size: int = 8, verbose_level: int = 0, sample_feature_length: int = 256):

        # Initialize class attributes
        self.data_dir = Path(data_dir_)
        self.data = []
        self.data_index = 0
        self.vis = Visualizer()
        self.batch_size = batch_size
        self.verbose_level = verbose_level
        self.sample_feature_length = sample_feature_length

        logger.info("Loading Data...")

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

        # Get number of batches, cut data to whole batches only
        self.batches = len(self.data) // self.batch_size
        self.data = self.data[: self.batches * self.batch_size]

        # Initial data shuffle
        self.shuffle()
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

    def next_batch(self):

        # Collate data; Formats N X to NHWC, N Y to NL

        # Accumulate batch data
        x_, x_lengths, y_, y_lengths = [], [], [], []
        for iter_1 in range(self.batch_size):
            next_x, next_x_length, next_y, next_y_length = self.next_item()

            x_.append(next_x)
            x_lengths.append(next_x_length)
            y_.append(next_y)
            y_lengths.append(next_y_length)

        # Find maximum label length
        max_label_len = max(y_lengths)

        # Pad all labels to maximum label length
        for iter_1 in range(self.batch_size):
            y_length_difference = max_label_len - y_lengths[iter_1]
            y_pad = tf.constant([[0, y_length_difference.numpy()], [0, 0]])
            y_[iter_1] = tf.pad(y_[iter_1], y_pad, constant_values=char_dict['_'])

        # Stack X, X lengths, Y, Y lengths into batch
        x_ = tf.stack(x_)
        x_lengths = tf.stack(x_lengths)
        y_ = tf.stack(y_)
        y_lengths = tf.stack(y_lengths)

        return x_, x_lengths, y_, y_lengths

    def next_item(self):
        x_, x_length, y_, y_length = self.get_current_raw_data(self.data_index)
        self.increment_index()
        return x_, x_length, y_, y_length

    def increment_index(self):
        self.data_index += 1
        if self.data_index > len(self):
            self.shuffle()
            self.data_index = 0
        return

    def __getitem__(self, item):
        return self.get_current_raw_data(item)

    def get_current_raw_data(self, index):

        # Constants observed from LibriSpeech Dataset
        sample_min_freq = 0
        sample_max_freq = 8000

        # Get current data pair
        current_data = self.data[index]
        x_path, y_ = current_data['x'], current_data['y']

        # Read in X
        x_data, sample_rate = soundfile.read(x_path)

        # Visualize if there is a high enough verbose level
        self.visualize(x_data, sample_rate, y_)

        # Process X and Y into raw forms
        array_frequencies, segment_times, x_ = signal.spectrogram(x_data, sample_rate, scaling='spectrum')
        y_ = char_str_to_number_seq(y_)

        # Convert to Tensor
        x_ = tf.convert_to_tensor(x_, dtype=tf.float32)
        y_ = tf.convert_to_tensor(y_, dtype=tf.int8)

        # Get lengths of X and Y, needed for CTCLoss
        x_length = tf.convert_to_tensor(x_.shape[1], dtype=tf.int16)
        y_length = tf.convert_to_tensor(y_.shape[0], dtype=tf.int16)

        # Scale X to standard feature size https://www.tensorflow.org/io/tutorials/audio
        x_ = tfio.experimental.audio.melscale(x_, sample_rate, self.sample_feature_length,
                                              fmin=sample_min_freq, fmax=sample_max_freq)

        # Expand X to HWC
        x_ = tf.expand_dims(x_, -1)

        return x_, x_length, y_, y_length

    @staticmethod
    def linear_normalization(data):
        return (data - data.min()) / (data.max() - data.min())

    def visualize(self, x_data, sample_rate, y_):
        # Use visualization class to show X, Y if verbose level is high enough
        if self.verbose_level > 1:
            self.vis.visualize_audio(x_data, sample_rate, y_)
        return


class Visualizer:

    @staticmethod
    def visualize_audio(x_data, sample_rate, y_):

        # Selected matplotlib visualization technique
        visualization_method = "pcolormesh"

        # Create matplotlib figure, axis
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Visualize data based on matplotlib pcolormesh or specgram optionally
        if visualization_method == "pcolormesh":
            array_frequencies, segment_times, x_ = signal.spectrogram(x_data, sample_rate, scaling='spectrum')
            ax.pcolormesh(segment_times, array_frequencies, x_)
        elif visualization_method == "specgram":
            ax.specgram(x_data, Fs=sample_rate)

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


