import os
from os.path import join
import librosa
from tqdm import tqdm

from .utils import Utilities


class MusicProcessor:
    def __init__(self):
        self.logger = Utilities.create_logger()
        self.files = [file for file in os.listdir(r"data\music")]

    def process(self):
        music_metadata = {}
        for file in tqdm(self.files):
            self.logger.info(f"Processing audio file {file}.")

            waveform, sampling_rate = librosa.load(path=join(r"data\music", file))
            # isfile(join(r"data\music", file))
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)

            music_metadata[file] = (waveform, sampling_rate, tempo, beat_frames)

            print(f"Estimated tempo: {tempo} beats per minute")

            # beat_times = librosa.frames_to_time(frames=beat_frames, sr=sampling_rate)

        print(music_metadata)
