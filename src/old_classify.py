import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class OldAudioClassifier:
    def __init__(self, audio_metadata: dict):
        self.scaler = StandardScaler()
        self.enriched_metadata = audio_metadata.copy()
        # These functions add 'function' and 'mood' to the metadata
        self._classify_function()
        self._classify_mood()

    def print_features(self):
        for name in self.enriched_metadata.keys():
            print(
                f"{name}: {self.enriched_metadata[name]['mood']}, {self.enriched_metadata[name]['function']}\n"
            )

    def get_features(self):
        features = {}
        for name in self.enriched_metadata.keys():
            features[name] = [
                self.enriched_metadata[name]["mood"],
                self.enriched_metadata[name]["function"],
            ]
        return features

    def _classify_mood(self):
        feature_vectors = []
        logger.info("Enriching audio metadata with moods")
        for data in tqdm(self.enriched_metadata.values()):
            feature_vectors.append(self._get_selected_features(data))

        X = self.scaler.fit_transform(feature_vectors)

        gmm = GaussianMixture(n_components=6, random_state=42, covariance_type="full")
        clusters = gmm.fit_predict(X)

        mood_mapping = self._map_clusters_to_moods(gmm.means_, feature_vectors)
        for name, cluster in tqdm(zip(self.enriched_metadata.keys(), clusters)):
            data = self.enriched_metadata[name]

            mood = mood_mapping[cluster]

            self.enriched_metadata[name]["mood"] = mood

    def _classify_function(self):
        logger.info("Enriching audio metadata with functions")
        # functions = {
        #     "background": lambda x: (
        #         0.3 < x["energy_mean"] < 0.6
        #         and 100 < x["tempo"] < 140
        #         and x["complexity_score"] < 28  # Not too complex
        #     ),
        #     "boss_battle": lambda x: (
        #         x["energy_mean"] > 0.7
        #         and x["tempo"] > 130
        #         and x["complexity_score"] > 25  # Complex composition
        #     ),
        #     "victory": lambda x: (
        #         x["tonal_stability"] > 0.04  # More stable tonality
        #         and x["tempo"] > 110
        #         and "Major" in x["key"]  # Victory themes often in major key
        #         and len(x["beat_times"]) < 50  # Shorter duration
        #     ),
        #     "game_over": lambda x: (
        #         x["tonal_stability"] < 0.03  # Less stable tonality
        #         and "Minor" in x["key"]  # Often in minor key
        #         and len(x["beat_times"]) < 20  # Very short duration
        #     ),
        #     "sound_effect": lambda x: (
        #         len(x["beat_times"]) < 10  # Very short duration
        #         and x["energy_mean"] > 0.1  # Noticeable volume
        #         and x["zero_crossing_rate_mean"] > 0.02  # More "noisy"
        #     )
        # }

        for name in tqdm(self.enriched_metadata.keys()):
            track_functions = set()  # Default
            if "complete" in name.lower():
                track_functions.add("victory")
            if "game_over" in name.lower() or "lost_life" in name.lower():
                track_functions.add("game_over")
            if "theme" in name.lower():
                track_functions.add("background")
            if "effect" in name.lower():
                track_functions.add("effect")

            # if not track_functions:
            #     track_functions.add("background")

            self.enriched_metadata[name]["function"] = track_functions

    def _get_selected_features(self, track_data):
        return np.array(
            [
                track_data["tempo"],  # 0
                track_data["spectral_centroid_mean"],  # 1
                track_data["energy_mean"],  # 2
                track_data["zero_crossing_rate_mean"],  # 3
                track_data["low_mfcc"],  # 4
                track_data["mid_mfcc"],  # 5
                track_data["high_mfcc"],  # 6
                track_data["mfcc_spread"],  # 7
                track_data["bass_contrast"],  # 8
                track_data["treble_contrast"],  # 9
                track_data["complexity_score"],  # 10
                track_data["tonal_stability"],  # 11
                track_data["rhythm_regularity"],  # 12
            ]
        )

    def _map_clusters_to_moods(self, cluster_features, feature_vectors):
        mood_mapping = {}

        all_features = np.array(feature_vectors)
        feature_means = np.mean(all_features, axis=0)
        feature_stds = np.std(all_features, axis=0)

        for i, features in enumerate(cluster_features):
            tempo = features[0]
            # spectral_centroid = features[1]
            energy = features[2]
            complexity = features[10]
            tonal_stability = features[11]

            # More nuanced mood classification using multiple features
            if energy > feature_means[2] + feature_stds[2] and tempo > 140:
                mood = "intense"
            elif (
                energy > feature_means[2]
                and tempo > 120
                and complexity > feature_means[10]
            ):
                mood = "energetic"
            elif (
                tonal_stability > feature_means[11] + feature_stds[11]
                and energy > feature_means[2]
            ):
                mood = "triumphant"
            elif (
                energy < feature_means[2] - feature_stds[2]
                and complexity < feature_means[10]
            ):
                mood = "mysterious"
            else:
                mood = "balanced"

            mood_mapping[i] = mood

        return mood_mapping
