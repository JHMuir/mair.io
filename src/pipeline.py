from .process import AudioProcessor
from .classify import AudioClassifier


class AudioPipeline:
    def __init__(self, audio_files: list):
        self.processor = AudioProcessor(audio_files=audio_files)
        self.classifier = AudioClassifier(audio_metadata=self.processor.audio_metadata)
