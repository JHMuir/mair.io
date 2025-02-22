from .process import AudioProcessor
from .classify import AudioClassifier


class AudioPipeline:
    def __init__(self, audio_files: list):
        self.processor = AudioProcessor(audio_files=audio_files)
        self.classifier = AudioClassifier(
            audio_metadata=self.processor.audio_metadata,
            metadata_averages=self.processor.metadata_averages,
        )

    def create_metadata_json(self, path: str = "audio_metadata.json") -> None:
        pass

    def _create_text_description(self, name, metadata):
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
