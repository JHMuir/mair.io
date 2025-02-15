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
            chromagram = librosa.feature.chroma_cqt(
                y=waveform, sr=sampling_rate, bins_per_octave=24
            )
            chroma_mean = np.mean(chromagram, axis=1).tolist()
            key = self._detect_key(chromagram=chromagram)

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
                "key": key,
                "complexity_score": complexity_score,
                "tonal_stability": tonal_stability,
            }
        self._logger.info("Audio feature extraction complete.")
        return audio_metadata

    def _detect_key(self, chromagram):
        chroma_vals = [np.sum(chromagram[i]) for i in range(12)]
        pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_freq = {pitches[i]: chroma_vals[i] for i in range(12)}
        keys = [pitches[i] + " Major" for i in range(12)] + [
            pitches[i] + " Minor" for i in range(12)
        ]

        # Krumhansl-Schmuckler key profiles
        major_profile = [
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88,
        ]
        minor_profile = [
            6.33,
            2.68,
            3.52,
            5.38,
            2.60,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17,
        ]

        correlations_maj = []
        correlations_min = []

        for i in range(12):
            estimated_key = [key_freq.get(pitches[(i + m) % 12]) for m in range(12)]
            correlations_maj.append(
                round(np.corrcoef(major_profile, estimated_key)[1, 0], 3)
            )
            correlations_min.append(
                round(np.corrcoef(minor_profile, estimated_key)[1, 0], 3)
            )

        key_dict = {
            **{keys[i]: correlations_maj[i] for i in range(12)},
            **{keys[i + 12]: correlations_min[i] for i in range(12)},
        }

        key = max(key_dict, key=key_dict.get)
        return key
