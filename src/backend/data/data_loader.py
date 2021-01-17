
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

plt.ioff()

random.seed(0)
np.random.seed(0)


class SpeechDataset(object):

    def __init__(self, data_dir_: str, verbose_level: int = 0):
        self.data_dir = Path(data_dir_)
        self.data = []
        self.data_index = 0
        self.vis = Visualizer()

        self.verbose_level = verbose_level

        logger.info("Loading Data...")
        num_voices = len(list(self.data_dir.iterdir()))
        for voice_dir in tqdm(self.data_dir.iterdir(), total=num_voices):
            if voice_dir.is_dir():
                for recording_dir in voice_dir.iterdir():

                    sample_label = None
                    current_samples = {}

                    for sample_path in recording_dir.iterdir():
                        if sample_path.suffix == '.txt':
                            sample_label = sample_path
                        elif sample_path.suffix == ".flac":
                            current_samples[sample_path.stem] = sample_path

                    with open(str(sample_label), 'r') as current_labels:
                        for line in current_labels.readlines():
                            line_sample = line.split(" ")[0]
                            if line_sample in current_samples:
                                sample_file = current_samples[line_sample]
                                line_label = line[len(line_sample) + 1:]

                                self.data.append({'x': sample_file, 'y': line_label})

        self.shuffle()

        return

    def shuffle(self):
        random.shuffle(self.data)
        return

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def increment_index(self):
        self.data_index += 1
        if self.data_index > len(self):
            self.shuffle()
            self.data_index = 0
        return

    def get_current_raw_data(self):

        current_data = self.data[self.data_index]
        x_path, y_ = current_data['x'], current_data['y']
        x_data, sample_rate = soundfile.read(x_path)

        return x_data, sample_rate, y_

    def next(self):
        self.increment_index()
        x_data, sample_rate, y_ = self.get_current_raw_data()

        if self.verbose_level > 1:
            self.vis.visualize_audio(x_data, sample_rate, y_)

        array_frequencies, segment_times, x_ = signal.spectrogram(x_data, sample_rate, scaling='spectrum')

        return x_, y_


class Visualizer:

    @staticmethod
    def visualize_audio(x_data, sample_rate, y_):

        visualization_method = "pcolormesh"

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if visualization_method == "pcolormesh":
            array_frequencies, segment_times, x_ = signal.spectrogram(x_data, sample_rate, scaling='spectrum')
            ax.pcolormesh(segment_times, array_frequencies, x_)
        elif visualization_method == "specgram":
            ax.specgram(x_data, Fs=sample_rate)

        ax.set_title("\n".join(wrap(y_, 55)), pad=20)

        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        fig.subplots_adjust(top=0.8)
        plt.show()


if __name__ == "__main__":

    data_dir = "raw/Dev/"
    dataset = SpeechDataset(data_dir, 2)
    tmp = next(dataset)



