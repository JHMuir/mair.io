from sklearn.preprocessing import StandardScaler


class AudioClassifier:
    def __init__(self, audio_metadata, logger):
        self._logger = logger
        self.audio_metadata = audio_metadata
        self.scaler = StandardScaler()

    def extract_features(self, data):
        pass

    def classify_mood(self):
        # feature_vectors = {}
        self._logger.info("Classifying Keys")
        for name in self.audio_metadata.keys():
            print(f"{name}")
