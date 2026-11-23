"""The network of the paper's last code (src/imagine2touch/models/models.py at commit c2ea2c3)."""

from torch import nn

from .imagine2touch_network import TouchNetwork


class Imagine2TouchCNN(TouchNetwork):
    """Three small convolutions into the embedding of 5 and a decoder of 10 to the touch signal; it does not rebuild the image."""

    name = "cnn"

    def __init__(self, touch_decoder=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 5), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(5, 3),
            nn.Conv2d(4, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(3, 3),
            nn.Conv2d(8, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(), nn.Sigmoid(), nn.Linear(32, self.EMBEDDING))
        self.touch = nn.Sequential(nn.Linear(self.EMBEDDING, touch_decoder), nn.ReLU(), nn.Linear(touch_decoder, self.SIGNAL))

    def forward(self, images):
        return self.touch(self.encoder(images[:, None])), None
