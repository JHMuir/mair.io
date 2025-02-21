import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AudioClassifier:
    def __init__(self, audio_metadata: dict):
        self.scaler = StandardScaler()
        self.enriched_metadata = audio_metadata.copy()
        self.global_averages = self._compute_global_averages()
        # These functions add 'function' and 'mood' to the metadata
        self._classify_function()
        self._classify_mood()

    def print_features(self) -> None:
        for name in self.enriched_metadata.keys():
            print(
                f"{name}: (mood: {self.enriched_metadata[name]['mood']}, function: {self.enriched_metadata[name]['function']})\n"
            )
        print(f"Global Averages: {self.global_averages}")

    def get_features(self) -> dict:
        features = {}
        for name in self.enriched_metadata.keys():
            features[name] = [
                self.enriched_metadata[name]["mood"],
                self.enriched_metadata[name]["function"],
            ]
        return features

    def _compute_global_averages(self) -> dict:
        energies = []
        tempos = []
        complexities = []
        tonal_stabilities = []

        for data in self.enriched_metadata.values():
            energies.append(data["energy_mean"])
            tempos.append(data["tempo"])
            complexities.append(data["complexity_score"])
            tonal_stabilities.append(data["tonal_stability"])

        global_averages = {}
        global_averages["energy"] = np.mean(energies)
        global_averages["tempo"] = np.mean(tempos)
        global_averages["complexity"] = np.mean(complexities)
        global_averages["tonal_stability"] = np.mean(tonal_stabilities)

        return global_averages

    def _classify_mood(self) -> None:
        for data in tqdm(self.enriched_metadata.values()):
            energy = data["energy_mean"]
            tempo = data["tempo"]
            complexity = data["complexity_score"]
            tonal = data["tonal_stability"]

            if (
                energy > self.global_averages["energy"]
                and tempo > self.global_averages["tempo"]
                and complexity > self.global_averages["complexity"]
            ):
                mood = "energetic"
            elif (
                tonal > self.global_averages["tonal_stability"]
                and energy > self.global_averages["energy"]
            ):
                mood = "triumphant"
            elif (
                energy < self.global_averages["energy"]
                and complexity < self.global_averages["complexity"]
            ):
                mood = "mysterious"
            else:
                mood = "balanced"
            data["mood"] = mood

    def _classify_function(self) -> None:
        logger.info("Enriching audio metadata with functions")

        for name in tqdm(self.enriched_metadata.keys()):
            audio_functions = []  # Default
            if "complete" in name.lower():
                audio_functions.append("victory")
            if "game_over" in name.lower() or "lost_life" in name.lower():
                audio_functions.append("game_over")
            if "theme" in name.lower():
                audio_functions.append("background")
            if "effect" in name.lower():
                audio_functions.append("effect")
            if "hurry" in name.lower():
                audio_functions.append("hurry")

            self.enriched_metadata[name]["function"] = ", ".join(audio_functions)
