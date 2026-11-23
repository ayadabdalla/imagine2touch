"""The network of the paper (Sec. II) and of the thesis (Fig. 8, vision to touch)."""

from torch import nn

from .imagine2touch_network import TouchNetwork


class Imagine2TouchMLP(TouchNetwork):
    """
    Two layers of 200 into the embedding of 5; from there a decoder of 500 to the touch signal and
    a second decoder of 2000 that rebuilds the image.  Small on purpose: the dataset has a few
    hundred touches per object.  Layers and starting weights (Kaiming) follow the training code of
    the thesis; the thesis's Fig. 8 draws the encoder as one layer of 200, the code has two.
    """

    name = "mlp"

    def __init__(self, hidden=200, touch_decoder=500, image_decoder=2000):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(self.PIXELS, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, self.EMBEDDING))
        self.touch = nn.Sequential(nn.Linear(self.EMBEDDING, touch_decoder), nn.ReLU(), nn.Linear(touch_decoder, self.SIGNAL))
        self.image = nn.Sequential(nn.Linear(self.EMBEDDING, image_decoder), nn.ReLU(), nn.Linear(image_decoder, self.PIXELS))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        embedding = self.encoder(images)
        return self.touch(embedding), self.image(embedding)
