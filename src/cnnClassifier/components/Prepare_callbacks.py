import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time

class PrepareCallbacksConfig:
    def __init__(self, tensorboard_root_log_dir, checkpoint_model_filepath):
        self.tensorboard_root_log_dir = tensorboard_root_log_dir
        self.checkpoint_model_filepath = checkpoint_model_filepath

class ConfigurationManager:
    def get_prepare_callback_config(self):
        return PrepareCallbacksConfig(
            tensorboard_root_log_dir="path/to/logs",
            checkpoint_model_filepath="artifacts/prepare_callbacks/checkpoint_dir/model.keras"  # Ensure this ends with .keras
        )

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        # Ensure the checkpoint file path ends with .keras
        checkpoint_filepath = self.config.checkpoint_model_filepath
        if not str(checkpoint_filepath).endswith('.keras'):
            raise ValueError(
                "The filepath provided must end in `.keras` (Keras model format). "
                f"Received: filepath={checkpoint_filepath}"
            )

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
        ]
