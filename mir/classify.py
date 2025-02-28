import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioClassifier:
    def __init__(self, audio_metadata: dict, metadata_averages: dict):
        self.moods = self._classify_mood(
            audio_metadata=audio_metadata,
            averages=metadata_averages,
        )
        self.in_game_functions = self._classify_function(audio_metadata=audio_metadata)
        self.classified_features = {
            name: [self.moods[name], self.in_game_functions[name]]
            for name in audio_metadata.keys()
        }

    def print_features(self) -> None:
        for name in self.classified_features.keys():
            print(
                f"{name}: (mood: {self.moods[name]}, function: {self.in_game_functions[name]})\n"
            )

    def _classify_mood(self, audio_metadata: dict, averages: dict) -> None:
        mood_dict = {}
        for name in tqdm(audio_metadata.keys()):
            energy = audio_metadata[name]["energy_mean"]
            tempo = audio_metadata[name]["tempo"]
            complexity = audio_metadata[name]["complexity_score"]
            tonal = audio_metadata[name]["tonal_stability"]

            if (
                energy > averages["energy_mean"]
                and tempo > averages["tempo"]
                and complexity > averages["complexity_score"]
            ):
                mood = "energetic"
            elif (
                tonal > averages["tonal_stability"] and energy > averages["energy_mean"]
            ):
                mood = "triumphant"
            elif (
                energy < averages["energy_mean"]
                and complexity < averages["complexity_score"]
            ):
                mood = "mysterious"
            else:
                mood = "balanced"
            # Adding the mood value to our original audio metadata
            audio_metadata[name]["mood"] = mood
            # Adding the mood value to AudioClassifier's mood dictionary
            mood_dict[name] = mood
        return mood_dict

    def _classify_function(self, audio_metadata: dict) -> None:
        logger.info("Enriching audio metadata with functions")
        in_game_functions = {}
        for name in tqdm(audio_metadata.keys()):
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

            audio_metadata[name]["function"] = ",".join(audio_functions)
            in_game_functions[name] = ",".join(audio_functions)
        return in_game_functions
