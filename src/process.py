from os.path import join
import logging
import numpy as np
import librosa
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, audio_files: list[str]):
        self.audio_metadata = self._create_metadata(audio_files=audio_files)

    def print_metadata(self) -> None:
        for name in self.audio_metadata.keys():
            logger.info(f"Printing features for {name}")
            for feature_name, value in self.audio_metadata[name].items():
                if isinstance(value, list):
                    print(f"{feature_name}: {len(value)} values")
                    if len(value) < 5:
                        print(f"Values: {value}")
                else:
                    print(f"{feature_name}: {value}")

    def _create_metadata(self, audio_files) -> dict:
        audio_metadata = {}
        logger.info("Processing audio tracks and extracting features.")
        for file in tqdm(audio_files):
            # Extracting features from audio file
            waveform, sampling_rate = librosa.load(path=join(r"data\music", file))

            # Tempo and Beat Information
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
            beat_times = librosa.frames_to_time(
                frames=beat_frames, sr=sampling_rate
            ).tolist()
            beat_strength = len(beat_times) / (
                tempo / 60
            )  # BPM relative to track length

            # Rhythm Patterns and Structure
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sampling_rate)
            tempo_scores = librosa.feature.rhythm.tempo(
                onset_envelope=onset_env, sr=sampling_rate, aggregate=None
            ).tolist()
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env, sr=sampling_rate
            )
            tempo_structure = np.mean(tempogram, axis=1).tolist()
            rhythm_regularity = np.std(tempo_structure) / np.mean(tempo_structure)

            # Spectral Features and Contrast
            spectral_centroids = librosa.feature.spectral_centroid(
                y=waveform, sr=sampling_rate
            )
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            spectral_contrast = librosa.feature.spectral_contrast(
                y=waveform, sr=sampling_rate
            )
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()
            bass_contrast = np.mean(spectral_contrast_mean[:3])
            treble_contrast = np.mean(spectral_contrast_mean[3:])

            # Energy/RMS
            energy = librosa.feature.rms(y=waveform)[0]
            energy_mean = float(np.mean(energy))
            energy_std = float(np.std(energy))

            # Zero Crossing Rate (Noisiness)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform)
            zero_crossing_rate_mean = float(np.mean(zero_crossing_rate))

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=waveform, sr=sampling_rate, n_mfcc=13)
            mfcc_profile = np.mean(mfccs, axis=1).tolist()
            low_mfcc = np.mean(mfcc_profile[:4])
            mid_mfcc = np.mean(mfcc_profile[4:9])
            high_mfcc = np.mean(mfcc_profile[9:])
            mfcc_spread = np.std(mfcc_profile)

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
                "tempo": int(tempo),
                "beat_times": beat_times,
                "beat_strength": beat_strength,
                "rhythm_regularity": rhythm_regularity,
                "tempo_scores": tempo_scores,
                "tempo_structure": tempo_structure,
                "spectral_centroid_mean": spectral_centroid_mean,
                "spectral_contrast_mean": spectral_contrast_mean,
                "bass_contrast": bass_contrast,
                "treble_contrast": treble_contrast,
                "chroma_mean": chroma_mean,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "zero_crossing_rate_mean": zero_crossing_rate_mean,
                "mfcc_profile": mfcc_profile,
                "low_mfcc": low_mfcc,
                "mid_mfcc": mid_mfcc,
                "high_mfcc": high_mfcc,
                "mfcc_spread": mfcc_spread,
                "tonal_features": tonnetz_mean,
                "key": key,
                "complexity_score": complexity_score,
                "tonal_stability": tonal_stability,
            }
        logger.info("Audio feature extraction complete.")
        return audio_metadata

    def _detect_key(self, chromagram) -> str:
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
