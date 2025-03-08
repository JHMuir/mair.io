import os
import json
import logging
from typing import Optional
from pydantic import ValidationError
from mir.process import AudioProcessor
from mir.classify import AudioClassifier
from mir.metadata_model import AudioMetadata, AudioMetadataCollection

logger = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(self, audio_files: list):
        logger.info("Initializing AudioPipeline")
        self.default_metadata_path = r"data\metadata\audio_metadata.json"
        self.metadata_collection: Optional[AudioMetadataCollection] = None
        self.audio_files = audio_files

        if os.path.exists(self.default_metadata_path):
            logger.info(
                f"Cached metadata found at {self.default_metadata_path}, loading from file"
            )
            audio_metadata = self._load_metadata_from_file()
            logger.info(
                "Initializing AudioPipeline and AudioClassifier with cached metadata"
            )
            self.processor = self._create_processor_from_cache(
                audio_metadata=audio_metadata
            )
            self.classifier = self._create_classifier_from_cache(
                audio_metadata=audio_metadata
            )
            self.metadata_collection = self._generate_validated_metadata(
                audio_metadata=audio_metadata
            )
        else:
            logger.info("Cached metadata not found, generating new metadata")
            self.processor = AudioProcessor(audio_files=audio_files)
            self.classifier = AudioClassifier(
                audio_metadata=self.processor.audio_metadata,
                metadata_averages=self.processor.metadata_averages,
            )
            self.metadata_collection = self._generate_validated_metadata(
                audio_metadata=self.processor.audio_metadata
            )
            logger.info(
                "Initialized AudioPipeline and AudioClassifier with new metadata"
            )

    def create_metadata_json(self, path: str = r"data\metadata\audio_metadata.json"):
        if os.path.exists(path):
            logger.info(f"Using cached metadata file at {path}")
            return path
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if self.metadata_collection:
                metadata_dict = {
                    name: model.model_dump()
                    for name, model in self.metadata_collection.root.items()
                }
                with open(path, "w") as f:
                    json.dump(metadata_dict, f, indent=4)
                logger.info(f"Metadata generated at {path}")
                return path
            else:
                logger.error("No validated metadata collection available")
                return ""

    def _generate_validated_metadata(
        self, audio_metadata: dict
    ) -> AudioMetadataCollection:
        logger.info("Generating annotated, validated metadata")
        pydantic_data = {}
        for name, data in audio_metadata.items():
            model_data = {}
            for key, value in data.items():
                if key in [
                    "waveform",
                    "tempo_scores",
                    "tempo_structure",
                    "chroma_mean",
                    "beat_times",
                ]:
                    continue
                model_data[key] = value
            model_data["description"] = self._create_text_description(
                name=name, metadata=data
            )

            try:
                track_model = AudioMetadata(**model_data)
                pydantic_data[name] = track_model
            except ValidationError as e:
                logger.error(f"Validation error for track {name}: {e}")
        return AudioMetadataCollection(root=pydantic_data)

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
