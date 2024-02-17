from src.train_waldo.model import trained_model


class Model:
    model = None

    def __init__(self):
        if not self.model:
            self.model = trained_model()

    def predict(self, image):
        return self.model.predict(image)
