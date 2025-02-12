import librosa
from pathlib import Path


class MusicProcessor:
    def __init__(self):
        self.file = librosa.load(path=Path(r"data\music\1.Ground_Theme.mp3"))

    def process(self):
        waveform, sampling_rate = librosa.load(self.file)

        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)

        print(f"Estimated tempo: {tempo} beats per minute")

        beat_times = librosa.frames_to_time(frames=beat_frames, sr=sampling_rate)

        print(beat_times)
