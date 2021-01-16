
from pathlib import Path
from tqdm import tqdm
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


class SpeechDataset(object):

    def __init__(self, data_dir_: str):
        self.data_dir = Path(data_dir_)
        self.data = []

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

        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        return


if __name__ == "__main__":

    data_dir = "raw/Train/"
    dataset = SpeechDataset(data_dir)
