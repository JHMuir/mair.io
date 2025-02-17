import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AudioClassifier:
    def __init__(self, audio_metadata):
        self.audio_metadata = audio_metadata
        self.scaler = StandardScaler()

    def extract_features(self, data):
        pass

    def classify_mood(self):
        # feature_vectors = {}
        logger.info("Classifying Keys")
        for name in self.audio_metadata.keys():
            print(f"{name}")
