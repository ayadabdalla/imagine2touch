"""A touch network with everything around it: scaling, training, saving."""

import copy

import numpy as np
import torch
from torch import nn

from imagine2touch.method.preprocessing import SIGNAL_SCALERS


class TouchModel:
    """
        model = TouchModel("mlp").fit(images, signals, validation=(images_v, signals_v))
        predicted = model.predict(images)        # (n, 15), in the units of the signals

    Training minimises the squared error of the scaled signal, plus `image_weight` times the error of
    the rebuilt image for networks that rebuild it: the binary cross-entropy for masks
    (`binary_images`, as in the thesis) or the squared error.  With a validation set, training stops
    when its error has not improved for `patience` epochs, and the best weights are kept.
    """

    def __init__(self, architecture="mlp", scaling="standard", epochs=300, lr=5e-4, batch_size=128,
                 image_weight=1.0, patience=30, seed=0, device="cpu", binary_images=False):
        from imagine2touch.method.models import ARCHITECTURES
        self.architecture, self.scaling, self.binary_images = architecture, scaling, binary_images
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.image_weight, self.patience, self.seed, self.device = image_weight, patience, seed, device
        self.scaler = SIGNAL_SCALERS[scaling]()
        torch.manual_seed(seed)
        self.network = ARCHITECTURES[architecture]().to(device)
        self.epochs_trained = 0

    # ------------------------------------------------------------------ training
    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array, np.float32), device=self.device)

    def _loss(self, images, targets):
        signal, rebuilt = self.network(images)
        loss = nn.functional.mse_loss(signal, targets)
        if rebuilt is not None and self.image_weight:
            image_loss = nn.functional.binary_cross_entropy_with_logits if self.binary_images else nn.functional.mse_loss
            loss = loss + self.image_weight * image_loss(rebuilt, images.flatten(1))
        return loss

    def target(self, signals):
        """The signals as the network is trained on them: the size of every value, as in the thesis's code."""
        return np.abs(np.asarray(signals, float))

    def fit(self, images, signals, validation=None):
        signals = self.target(signals)
        self.scaler.fit(signals)
        X, Y = self._tensor(images), self._tensor(self.scaler.transform(signals))
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.epochs):
            self.network.train()
            for batch in torch.randperm(len(X), generator=generator).split(self.batch_size):
                if len(batch) < 2:
                    continue
                loss = self._loss(X[batch], Y[batch])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            self.epochs_trained = epoch + 1
            if validation is not None:
                error = float(((self.predict(validation[0]) - self.target(validation[1])) ** 2).mean())
                if error < best:
                    best, best_state, waited = error, copy.deepcopy(self.network.state_dict()), 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ use
    def predict(self, images):
        self.network.eval()
        with torch.no_grad():
            scaled = torch.cat([self.network(batch)[0] for batch in self._tensor(images).split(1024)])
        return self.scaler.inverse(scaled.cpu().numpy())

    def standardise(self, signals):
        """Signals in the scaled form of the training targets, in which a felt and a predicted one are compared."""
        return self.scaler.comparable(self.target(signals))

    # ------------------------------------------------------------------ saving
    def save(self, path):
        torch.save(dict(network=self.network.state_dict(), scaler=vars(self.scaler),
                        architecture=self.architecture, scaling=self.scaling, binary_images=self.binary_images), path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(state["architecture"], state["scaling"], device=device, binary_images=state["binary_images"])
        model.network.load_state_dict(state["network"])
        vars(model.scaler).update(state["scaler"])
        return model
