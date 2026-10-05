import json

import numpy as np
from keras import Input
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredError
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import set_random_seed
from sklearn.linear_model import LogisticRegression

from . import config


class Autoencoder:
    """Autoencoder over edge features, trained on real edges only.

    Two ways to score a pair:
      link_scores - minus the reconstruction error (real-looking edges rebuild well)
      head_scores - logistic regression on the 32-dim latent code (needs fit_head)
    """

    def __init__(self, model, history=None):
        self.model = model
        self.history = history
        self.encoder = Model(model.input, model.get_layer("latent").output)
        self.head = None

    @classmethod
    def build(cls, input_dim):
        """input -> 64 -> dropout -> 32 (latent) -> 64 -> input"""
        set_random_seed(config.SEED)
        inputs = Input(shape=(input_dim,))
        x = Dense(config.HIDDEN_DIM, activation="relu")(inputs)
        x = Dropout(config.DROPOUT)(x)
        latent = Dense(config.LATENT_DIM, activation="relu", name="latent")(x)
        x = Dense(config.HIDDEN_DIM, activation="relu")(latent)
        outputs = Dense(input_dim)(x)

        model = Model(inputs, outputs, name="autoencoder")
        model.compile(optimizer=Adam(config.LR), loss=MeanSquaredError())
        return cls(model)

    def fit(self, x_train, x_val):
        """Learn to rebuild the input (x -> x)."""
        history = self.model.fit(
            x_train, x_train,
            validation_data=(x_val, x_val),
            epochs=config.TRAINING_EPOCHS,
            batch_size=config.TRAIN_BATCH_SIZE,
        )
        self.history = history.history
        loss, val_loss = self.history["loss"], self.history["val_loss"]
        print(f"[Autoencoder.fit]: {len(loss)} epochs, loss {loss[0]:.4f} -> {loss[-1]:.4f}, "
              f"val_loss {val_loss[0]:.4f} -> {val_loss[-1]:.4f}")
        return history

    def reconstruction_error(self, x):
        """Mean squared error per row."""
        x_hat = self.model.predict(x, verbose=0)
        return np.mean((x - x_hat) ** 2, axis=1)

    def link_scores(self, x):
        return -self.reconstruction_error(x)

    def fit_head(self, x_pos, x_neg):
        """Train the logistic regression on the latent codes: positives = 1, negatives = 0."""
        z = self.encoder.predict(np.vstack([x_pos, x_neg]), verbose=0)
        y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
        self.head = LogisticRegression().fit(z, y)

    def head_scores(self, x):
        """Probability that the pair is a real edge."""
        z = self.encoder.predict(x, verbose=0)
        return self.head.predict_proba(z)[:, 1]

    @staticmethod
    def file():
        return config.MODELS_DIR / f"autoencoder_{config.EDGE_OP}.keras"

    @staticmethod
    def history_file():
        return config.MODELS_DIR / f"autoencoder_{config.EDGE_OP}_history.json"

    def save(self):
        """Save the Keras model and its history. The logistic regression head is not saved."""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(self.file())
        self.history_file().write_text(json.dumps(self.history, indent=2))
        print(f"[Autoencoder.save]: {self.file().name}")

    @classmethod
    def load(cls):
        """Load the saved model. Call fit_head again before using head_scores."""
        history = None
        if cls.history_file().exists():
            history = json.loads(cls.history_file().read_text())
        print(f"[Autoencoder.load]: {cls.file().name}")
        return cls(load_model(cls.file()), history)
