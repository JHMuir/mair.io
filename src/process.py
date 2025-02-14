from os.path import join
import numpy as np
import librosa
from tqdm import tqdm


class AudioProcessor:
    def __init__(self, audio_files: list[str], logger):
        self._logger = logger
        self._audio_files = audio_files
        self._audio_metadata = self._create_metadata()

    def print_metadata(self) -> None:
        for file in self._audio_files:
            self._logger.info(f"Printing features for {file}")
            for feature_name, value in self._audio_metadata[file].items():
                if isinstance(value, list):
                    print(f"{feature_name}: {len(value)} values")
                    if len(value) < 5:
                        print(f"Values: {value}")
                else:
                    print(f"{feature_name}: {value}")

    def get_audio_metadata(self) -> dict:
        return self._audio_metadata

    def _create_metadata(self) -> dict:  # isfile(join(r"data\music", file))
        audio_metadata = {}
        self._logger.info("Processing audio tracks and extracting features.")
        for file in tqdm(self._audio_files):
            # Extracting features from audio file
            waveform, sampling_rate = librosa.load(path=join(r"data\music", file))

            # Tempo and Beat Information
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
            beat_times = librosa.frames_to_time(
                frames=beat_frames, sr=sampling_rate
            ).tolist()

            # Rhythm Patterns and Structure
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sampling_rate)
            tempo_scores = librosa.feature.rhythm.tempo(
                onset_envelope=onset_env, sr=sampling_rate, aggregate=None
            ).tolist()
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env, sr=sampling_rate
            )
            tempo_structure = np.mean(tempogram, axis=1).tolist()

            # Spectral Features and Contrast
            spectral_centroids = librosa.feature.spectral_centroid(
                y=waveform, sr=sampling_rate
            )
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            spectral_contrast = librosa.feature.spectral_contrast(
                y=waveform, sr=sampling_rate
            )
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()

            # Energy/RMS
            energy = librosa.feature.rms(y=waveform)[0]
            energy_mean = float(np.mean(energy))
            energy_std = float(np.std(energy))

            # Zero Crossing Rate (Noisiness)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform)
            zero_crossing_rate_mean = float(np.mean(zero_crossing_rate))

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=waveform, sr=sampling_rate, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1).tolist()

            # Tonal Features
            tonnetz = librosa.feature.tonnetz(y=waveform, sr=sampling_rate)
            tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

            # Chromagram (Harmony and Key)
            chromagram = librosa.feature.chroma_stft(y=waveform, sr=sampling_rate)
            chroma_mean = np.mean(chromagram, axis=1).tolist()
            # Detecting the key of the audio file using chromagram
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
            estimated_key_index = np.argmax(chroma_mean)
            estimated_key = chroma_to_key[estimated_key_index]

            complexity_score = float(np.mean(spectral_contrast_mean))
            tonal_stability = float(np.std(tonnetz_mean))

            # Adding features to a dictionary with the file name as key
            audio_metadata[file] = {
                "waveform": waveform,
                "sampling_rate": sampling_rate,
                "tempo": float(tempo),
                "beat_times": beat_times,
                "tempo_scores": tempo_scores,
                "tempo_structure": tempo_structure,
                "spectral_centroid_mean": spectral_centroid_mean,
                "chroma_mean": chroma_mean,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "zero_crossing_rate_mean": zero_crossing_rate_mean,
                "mfcc_profile": mfcc_means,
                "tonal_features": tonnetz_mean,
                "key": estimated_key,
                "complexity_score": complexity_score,
                "tonal_stability": tonal_stability,
            }
        self._logger.info("Audio feature extraction complete.")
        return audio_metadata
