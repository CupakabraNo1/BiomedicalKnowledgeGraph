"""Autoencoder for link prediction / anomaly detection.

Stub — to be implemented during the project.
"""
from __future__ import annotations


def build_autoencoder(input_dim: int):
    """Define and return an (uncompiled or compiled) Keras model.

    TODO: encoder -> latent layer -> decoder.
    """
    raise NotImplementedError


def reconstruction_error(model, X):
    """Per-sample score = reconstruction MSE (higher = more anomalous).

    TODO.
    """
    raise NotImplementedError
