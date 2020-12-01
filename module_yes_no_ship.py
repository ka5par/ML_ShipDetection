import fastai
from fastai.vision.all import *


class ShipConfirmer:
    MODEL_PATH = "output/ship_yes_no.pkl"

    @classmethod
    def confirm_ship_status(image_path):
        """
        confirm_ship_status takes image path and returns a confidence of the ship existing in the image.
        """
        learner = load_learner(MODEL_PATH)
        # Predicted class '0'/'1' is ship
        # Tensor of class
        # probabilities - tensor array of class probabilities
        predicted_class, tensor, probabilites = learner.predict(image_path)

        ship_probability = probabilites.numpy()[1]
        return ship_probability
