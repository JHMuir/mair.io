from sklearn.preprocessing import StandardScaler


class AudioClassifier:
    def __init__(self, audio_metadata):
        self.audio_metadata = audio_metadata
        self.scaler = StandardScaler()

    def do_a_thing(self):
        print("did a thing")
