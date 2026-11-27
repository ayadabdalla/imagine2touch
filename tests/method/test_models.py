"""The two touch networks, their training, and the classifier of the stamps."""

import numpy as np
import pytest

from imagine2touch.method.models import ARCHITECTURES, ShapeClassifier, TouchModel, TouchNetwork


def learnable(n=160, seed=0):
    """Images whose left and right halves are filled by different amounts, and signals that follow those amounts."""
    rng = np.random.default_rng(seed)
    left, right = rng.random(n), rng.random(n)
    images = np.zeros((n, 48, 48), np.float32)
    images[:, :, :24], images[:, :, 24:] = left[:, None, None], right[:, None, None]
    signal = np.outer(left, np.linspace(1, 15, 15)) + np.outer(right, np.linspace(15, 1, 15))
    return images, np.abs(signal + rng.normal(0, 0.05, signal.shape))        # a signal is the size of a change


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_model_learns_and_predicts_in_the_units_of_the_signal(architecture):
    images, signal = learnable()
    model = TouchModel(architecture, epochs=60, lr=2e-3, batch_size=32).fit(images[:120], signal[:120])
    predicted = model.predict(images[120:])
    assert predicted.shape == (40, 15)
    error = np.sqrt(((predicted - signal[120:]) ** 2).mean())
    baseline = np.sqrt(((signal[:120].mean(0) - signal[120:]) ** 2).mean())
    assert error < 0.5 * baseline


def test_training_stops_on_the_validation_set_and_reloads(tmp_path):
    images, signal = learnable()
    model = TouchModel("cnn", epochs=200, patience=3, lr=2e-3, batch_size=32)
    model.fit(images[:100], signal[:100], validation=(images[100:], np.random.default_rng(1).normal(size=(60, 15))))
    assert model.epochs_trained < 200                                         # noise cannot keep improving
    model.save(tmp_path / "model.pt")
    assert TouchModel.load(tmp_path / "model.pt").predict(images[:5]) == pytest.approx(model.predict(images[:5]), abs=1e-5)


def test_the_networks_share_one_interface():
    import torch
    for cls in ARCHITECTURES.values():
        network = cls()
        signal, rebuilt = network(torch.zeros(4, 48, 48))
        assert isinstance(network, TouchNetwork) and signal.shape == (4, 15)
        assert rebuilt is None or rebuilt.shape == (4, 48 * 48)


def test_the_classifier_names_the_stamps_and_reloads(stamps, tmp_path):
    train, test = stamps.role == "train", stamps.role == "test"
    model = ShapeClassifier("wide", epochs=150).fit(stamps.signal[train], stamps.stamp[train])
    p = model.probabilities(stamps.signal[test])
    assert p.shape == (test.sum(), 5) and p.sum(1) == pytest.approx(1, abs=1e-5)
    model.save(tmp_path / "classifier.pt")
    assert ShapeClassifier.load(tmp_path / "classifier.pt").probabilities(stamps.signal[test]) == pytest.approx(p, abs=1e-6)
    assert (p.argmax(1) == stamps.stamp[test]).mean() > 0.6                    # chance is 0.2; the paper's full training reaches 0.91
