from src.imagine2touch.reskin_sensor.sensor_proc import ReSkinProcess, ReSkinSettings
import time
import signal
import matplotlib.pyplot as plt
import numpy as np


# initialize sensor
sensor_settings = ReSkinSettings(
    num_mags=5, port="/dev/ttyACM0", baudrate=115200, burst_mode=True, device_id=1
)  # TODO :: take them as arguments
sensor_process = ReSkinProcess(sensor_settings)
sensor_process.start()
time.sleep(0.1)

# Add signal handler to be able to terminate using keyboard
end = False


def cb_int(*args):
    global end
    end = True


signal.signal(signal.SIGINT, cb_int)
reskin_recording = []
i = 0
while not end:
    i = i + 1
    reskin_recording.append(sensor_process.get_data(num_samples=1))
    if i % 10 == 0:
        reskin_reading = np.squeeze(reskin_recording)[
            :, 2
        ]  # extract lists of magnetometers values and temperatures as array of lists
        reskin_reading = list(
            reskin_reading
        )  # convert to list of lists then to nd array
        reskin_reading = np.asarray(reskin_reading, dtype=object)
        reskin_reading = np.delete(
            reskin_reading, [0, 4, 8, 12, 16], 1
        )  # eliminate temperatures
        reskin_reading = np.swapaxes(reskin_reading, 0, 1)
        print(reskin_reading.shape)
        print(np.linalg.norm(reskin_reading[:, -1], 2))
        print(reskin_reading[:, -1])
        # draw
        plt.plot(
            [
                "bx1",
                "by1",
                "bz1",
                "bx2",
                "by2",
                "bz2",
                "bx3",
                "by3",
                "bz3",
                "bx4",
                "by4",
                "bz4",
                "bx5",
                "by5",
                "bz5",
            ],
            reskin_reading[:, -1],
        )
        plt.draw()
        plt.pause(0.1)
