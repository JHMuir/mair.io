import os
from os.path import join
import numpy as np
import librosa
from tqdm import tqdm

from .utils import Utilities


class AudioProcessor:
    def __init__(self):
        self.logger = Utilities.create_logger()
        self.files = [file for file in os.listdir(r"data\music")]
        self.audio_metadata = self.create_metadata()

    def create_metadata(self) -> dict:  # isfile(join(r"data\music", file))
        audio_metadata = {}
        self.logger.info("Processing audio tracks and extracting features.")
        for file in tqdm(self.files):
            # Extracting features from audio file
            waveform, sampling_rate = librosa.load(path=join(r"data\music", file))
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
            beat_times = librosa.frames_to_time(
                frames=beat_frames, sr=sampling_rate
            ).tolist()
            spectral_centroids = librosa.feature.spectral_centroid(
                y=waveform, sr=sampling_rate
            )[0]
            chromagram = librosa.feature.chroma_stft(y=waveform, sr=sampling_rate)
            energy = librosa.feature.rms(y=waveform)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform)[0]

            # Detecting the key of the audio file using extracted features
            chroma_to_key = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            chroma_mean = np.mean(chromagram, axis=1).tolist()
            estimated_key_index = np.argmax(chroma_mean)
            estimated_key = chroma_to_key[estimated_key_index]

            # Adding features to a dictionary with the file name as key
            audio_metadata[file] = {
                "waveform": waveform,
                "sampling_rate": sampling_rate,
                "tempo": float(tempo),
                "beat_times": beat_times,
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "chroma_mean": chroma_mean,
                "energy_mean": float(np.mean(energy)),
                "energy_std": float(np.std(energy)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "key": estimated_key,
            }
        self.logger.info("Audio feature extraction complete.")
        return audio_metadata

    def print_features(self):
        for file in self.files:
            self.logger.info(f"Printing features for {file}")
            for feature_name, value in self.audio_metadata[file].items():
                if isinstance(value, list):
                    print(f"{feature_name}: {len(value)} values")
                    if len(value) < 5:  # Only print if the list is short
                        print(f"Values: {value}")
                else:
                    print(f"{feature_name}: {value}")
