"""
The method of the paper.  It knows nothing of where its inputs come from: recorded data and a
touching robot use it alike.

  preprocessing/    depth image -> contact mask; scaling of the touch signal
  models/           the touch model (image -> touch signal) and the shape classifier
  recognition/      naming an object from touches: evidence per touch, a rule over touches
"""
