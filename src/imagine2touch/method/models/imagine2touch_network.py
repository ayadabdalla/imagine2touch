"""What a network of the touch model has to be."""

from abc import ABC, abstractmethod

from torch import nn


class TouchNetwork(nn.Module, ABC):
    """
    Maps images (n, 48, 48) to scaled touch signals (n, 15) through a small embedding shared by
    vision and touch.  Networks that also rebuild their input from the embedding, to tie the
    embedding to the image, return it as logits (n, 48 * 48); the others return None in its place.
    """

    name: str
    PIXELS, SIGNAL, EMBEDDING = 48 * 48, 15, 5

    @abstractmethod
    def forward(self, images):
        """Returns (signal, rebuilt image or None)."""
