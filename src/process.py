import librosa
from .utils import Utilities


class MusicProcessor:
    def __init__(self):
        self.logger = Utilities.create_logger()
        self.logger.info("Loading audio file.")

    def process(self):
        waveform, sampling_rate = librosa.load(path=r"data\music\1.Ground_Theme.mp3")

        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)

        print(f"Estimated tempo: {tempo} beats per minute")

        beat_times = librosa.frames_to_time(frames=beat_frames, sr=sampling_rate)

        print(beat_times)
