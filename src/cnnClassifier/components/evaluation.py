import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import Evaluationconfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: Evaluationconfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        # Load the model from the specified path
        self.model = self.load_model(self.config.path_of_model)
        
        # Generate validation data
        self._valid_generator()
        
        # Evaluate the model on validation data
        self.score = self.model.evaluate(self.valid_generator)  # Use self.model

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
