"""How an approach of the sensor to a surface ends."""

from enum import IntEnum


class ApproachOutcome(IntEnum):
    """The values are the codes the robot writes to the recordings."""

    SENSOR = 2       # the touch reading jumped: the sensor felt the surface
    WRENCH = 3       # the robot's own force estimate passed its limit first
    AIR = 4          # the sensor passed the target without either
    ABORTED = -1     # the approach was refused or stopped for safety
    UNKNOWN = 0      # the earliest sessions did not record how an approach ended

    @property
    def label(self):
        return self.name.lower()

    @classmethod
    def label_of(cls, code):
        """The label of a recorded code; other codes of the robot's controller count as aborted."""
        return cls(code).label if code in cls._value2member_map_ else cls.ABORTED.label
