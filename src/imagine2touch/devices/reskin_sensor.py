"""
A ReSkin sensor behind the TouchSensor interface, through reskin_sensor
(https://github.com/raunaqbhirangi/reskin_sensor).

Written from the scripts that recorded the paper's data (commit c2ea2c3 of this repository) and not run on hardware since:
check it against your setup before letting the robot move.
"""

from imagine2touch.reskin_reading import ReSkinReading
from imagine2touch.interfaces import TouchSensor


class ReSkinSensor(TouchSensor):
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, device_id=1):
        from reskin_sensor import ReSkinProcess, ReSkinSettings      # installed by `pip install -e .[robot]`
        self.process = ReSkinProcess(ReSkinSettings(num_mags=5, port=port, baudrate=baudrate, burst_mode=True,
                                                    device_id=device_id))
        self.process.start()

    def read(self, samples=1):
        return ReSkinReading.mean_field(self.process.get_data(num_samples=samples))
