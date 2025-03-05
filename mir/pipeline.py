import json
from mir.process import AudioProcessor
from mir.classify import AudioClassifier


class AudioPipeline:
    def __init__(self, audio_files: list):
        self.processor = AudioProcessor(audio_files=audio_files)
        self.classifier = AudioClassifier(
            audio_metadata=self.processor.audio_metadata,
            metadata_averages=self.processor.metadata_averages,
        )

    def create_metadata_json(
        self, path: str = r"data\metadata\audio_metadata.json"
    ) -> str:
        processed_metadata = {}
        for name, data in self.processor.audio_metadata.items():
            track_data = {}
            for feature_name, value in data.items():
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

        return path

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

    def print_descriptions(self):
        for name, metadata in self.processor.audio_metadata.items():
            print(self._create_text_description(name=name, metadata=metadata))
