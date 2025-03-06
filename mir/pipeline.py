import os
import json
import logging
from mir.process import AudioProcessor
from mir.classify import AudioClassifier

logger = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(self, audio_files: list):
        self.default_metadata_path = r"data\metadata\audio_metadata.json"
        self.audio_files = audio_files

        if os.path.exists(self.default_metadata_path):
            logger.info(
                f"Cached metadata found at {self.default_metadata_path}, loading from file"
            )

            audio_metadata = self._load_metadata_from_file()
            self.processor = self._create_processor_from_cache(
                audio_metadata=audio_metadata
            )
            self.classifier = self._create_classifier_from_cache(
                audio_metadata=audio_metadata
            )
        else:
            logger.info("Cached metadata not found, generating new metadata")

            self.processor = AudioProcessor(audio_files=audio_files)
            self.classifier = AudioClassifier(
                audio_metadata=self.processor.audio_metadata,
                metadata_averages=self.processor.metadata_averages,
            )

    def create_metadata_json(
        self, path: str = r"data\metadata\audio_metadata.json"
    ) -> str:
        # Checking if metadata already exists
        if os.path.exists(path):
            logger.info(f"Using cached metadata file at {path}")
            return path
        else:
            processed_metadata = {}
            for name, data in self.processor.audio_metadata.items():
                track_data = {}
                for feature_name, value in data.items():
                    # TODO: add missing values from original audio_metadata to the json
                    if feature_name in ["waveform"]:
                        continue
                    if isinstance(value, (int, float, str)):
                        track_data[feature_name] = value

                track_data["description"] = self._create_text_description(
                    name=name, metadata=data
                )
                processed_metadata[name] = track_data

            with open(path, "w") as f:
                json.dump(processed_metadata, f, indent=4)

            logger.info(f"Metadata generated at {path}")
            return path

    def print_descriptions(self) -> None:
        for name, metadata in self.processor.audio_metadata.items():
            print(self._create_text_description(name=name, metadata=metadata))

    def _create_text_description(self, name, metadata) -> str:
        description = (
            f"This track is named {name}."
            f"This is a {metadata['mood']} {metadata['function']} track in {metadata['key']} "
            f"with a tempo of {metadata['tempo']} BPM. "
            f"It has {'high' if metadata['energy_mean'] > self.processor.metadata_averages['energy_mean'] else 'low'} energy "
            f"and {'complex' if metadata['complexity_score'] > self.processor.metadata_averages['complexity_score'] else 'simple'} structure. "
            f"The track features {'strong' if metadata['bass_contrast'] > self.processor.metadata_averages['bass_contrast'] else 'subtle'} bass "
            f"and {'bright' if metadata['treble_contrast'] > self.processor.metadata_averages['treble_contrast'] else 'warm'} treble characteristics.\n"
        )
        return description

    def _load_metadata_from_file(self) -> dict:
        # Load metadata from the file
        metadata = {}
        with open(self.default_metadata_path, "r") as f:
            metadata = json.load(f)

        # Convert the metadata in a structure synonmous with AudioProcessor.audio_metadata
        converted_metadata = {}
        for name, data in metadata.items():
            # Copy all fields except description (which is generated, not loaded)
            converted_metadata[name] = {
                k: v for k, v in data.items() if k != "description"
            }
            # Set waveform to None since we don't store it in the JSON
            converted_metadata[name]["waveform"] = None
        return converted_metadata

    def _create_processor_from_cache(self, audio_metadata: dict):
        # Empty intialization of AudioProcessor
        processor = AudioProcessor.__new__(AudioProcessor)

        # Filling in AudioProcessor's members from our cached metadata
        processor.audio_metadata = audio_metadata
        processor.metadata_averages = processor._create_metadata_averages(
            audio_metadata=audio_metadata
        )

        return processor

    def _create_classifier_from_cache(self, audio_metadata: dict):
        # Empty intialization of AudioClassifier
        classifier = AudioClassifier.__new__(AudioClassifier)

        # Filling in AudioClassifier's members from our cached metadata
        moods = {}
        functions = {}
        for name, data in audio_metadata.items():
            if "mood" in data:
                moods[name] = data["mood"]
            if "function" in data:
                functions[name] = data["function"]

        classifier.moods = moods
        classifier.in_game_functions = functions
        classifier.classified_features = {
            name: [moods[name], functions[name]] for name in audio_metadata.keys()
        }

        return classifier
