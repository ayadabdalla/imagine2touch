# Running on the robot

This guide takes you from an empty table to a robot that is calibrated, has scanned its objects and
collects touches of them.

None of this has been run on hardware since the paper. The calibration and scanning tools were
rewritten from the paper's code and have only been checked on the simulated robot. Treat your first
runs on a real arm as tests: move slowly and keep a hand on the stop button.

Scanning and collecting also have a `--simulate` option. Run the simulated version first. It needs
no hardware, only the dataset in `data/` (see the [README](../README.md#getting-started)).

## What you need

| | |
|---|---|
| Franka Emika Panda | the arm |
| [ReSkin](https://reskin.dev) on a [printed finger](../assets/finger/reskin_finger_short.stl) that the hand holds, with the burst-stream firmware of [reskin_sensor](https://github.com/raunaqbhirangi/reskin_sensor) on its board | the touch sensor |
| RealSense depth camera at the wrist | the depth image before each touch, and the scans |
| A printed board of AprilTags (printed in step 3) | calibrating the camera and marking the table |
| ROS on the machine that controls the arm | sending commands to the arm |

## 1. Install

On the machine that runs the scripts, install the repository with the drivers for the ReSkin and the
RealSense, and OpenCV:

```bash
pip install -e ".[robot]"
```

The scripts talk to the arm through [robot_io](https://github.com/lukashermann/robot_io) and its ROS
package `robot_io_ros`. On the machine that controls the arm, get the version this guide was written
for and apply the patch from this repository:

```bash
git clone https://github.com/lukashermann/robot_io && cd robot_io
git checkout 14cc2ae
git apply PATH/TO/imagine2touch/src/imagine2touch/devices/robot_io_server.patch
```

The patch turns the public server into the one the paper's setup ran. It makes four changes:

- It names the server's node `robot_io_ros_server`, the name the scripts connect to.
- It stops the server from reading the robot's state while a move is being commanded.
- It reports a move that failed as a failed call, so the scripts notice.
- It adds a blocking straight-line move, and publishes the joint states and the measured force as
  ROS topics.

Next, follow robot_io's own instructions to set up the Panda, build `robot_io_ros` in a catkin
workspace, and start the server:

```bash
roslaunch robot_io_ros panda_server.launch
```

Before starting it, set `hydra_config_path` in `robot_io_ros/launch/panda_server.launch` to the
`robot_io/conf` folder of your copy of robot_io, relative to `robot_io_ros/scripts/`.

The machine that runs the scripts needs the client side as well: install `robot_io`
(`pip install -e robot_io`), build `robot_io_ros` in a catkin workspace, and source that workspace.
On this machine the scripts and the ROS client must run in the same Python. The paper's setup used
ROS Noetic, which comes with Python 3.8, and this repository works on 3.8: its tests pass there, and
so do the simulated runs of the scripts below. The one thing that has not been tried is the scripts
importing the ROS client.

## 2. Tell the arm where the sensor is

The scripts move the sensor, not the flange. The tool frame that the server reports must therefore
be the middle of the sensing surface, with its z axis pointing out of the skin. Set this as the end
effector in Franka Desk.

On the paper's mount, the sensing surface was 126 mm along the flange's axis and turned by 45
degrees. That value is fitted from the paper's recordings, and it agrees with the geometry. The
Franka hand held a printed finger between its fingertips, whose middle is 103 mm from the flange.
The sensing surface was about 22 mm below that point: 18 mm of finger plus the thickness of the
sensor.

The finger is [assets/finger/reskin_finger_short.stl](../assets/finger/reskin_finger_short.stl). It
is 31 mm long, and the file is in centimetres.

## 3. Calibrate the camera and the table

```bash
python -m imagine2touch.procedures.scripts.calibrate_camera --print-board board.pdf
python -m imagine2touch.procedures.scripts.calibrate_camera --board-centre 0.45 0.0 0.0 --out calibration
```

Print the board with a tag size of 30 mm. Lay it flat where the objects will later stand.
`--board-centre` is the rough position of the board's middle in the frame of the robot's base, in
metres. Its third number is the height of the table.

For another tag size, pass the width of one tag in metres as `--tag`.

The robot then moves the camera to 13 places around the board, 15 cm from its middle: one straight
above it and twelve tilted to the sides. At each place it notes where it holds the sensor and where
the camera sees the board. To know where to move, it needs a rough pose of the camera relative to
the sensor. It uses the paper's calibration for that; for a different mount, pass your own with
`--expected`, a 4 x 4 `.npy` file.

The result does not build on that rough pose. It is computed from the 13 views alone: the board did
not move, so there is one place of the camera for which all views put the board at the same spot.
The script writes two files:

- `calibration/camera.npy`: the camera relative to the sensor, that is, in the tool frame of step 2,
- `calibration/world.npy`: the board relative to the robot's base.

From here on, the board's frame is the world frame. Its origin is at the lower left tag and its z
axis points up from the table.

The script also prints how far the views disagree about the board's position, which tells you how
good the calibration is.

## 4. Scan each object

```bash
python -m imagine2touch.procedures.scripts.scan_object --simulate --object star --out scans     # the same procedure, simulated
python -m imagine2touch.procedures.scripts.scan_object --object star --world calibration/world.npy --camera calibration/camera.npy --out scans
```

Remove the board and place the object at the origin of the world frame. The robot looks at it from
above and from four sides, then writes two files: `scans/star.pcd`, the scan, and `scans/star.npy`,
its surface normals.

The scan keeps only points more than one centimetre above the table, so that the table itself is
not part of it. A flat object would be cut off as well, so put it on a block. `--box` sets which part
of the world frame is kept.

## 5. Collect touches

```bash
python -m imagine2touch.procedures.scripts.collect_data --simulate --object star --touches 20 --out data/simulated
python -m imagine2touch.procedures.scripts.collect_data --object star_v_1 --scan scans/star.pcd --touches 300 --out data/recordings \
    --world calibration/world.npy --camera calibration/camera.npy
python -m imagine2touch.datasets.recordings.recording_reducer --source data/recordings --all --out data/my_touches.npz
```

`--object` is the name of the session. To record the same shape more than once, end the name with
`_v_<number>`: `star_v_1` and `star_v_2` are two sessions of the shape `star`.

The last command turns the recordings into a dataset with the same layout as the paper's. Always
pass `--out`. Without it, the command writes to `data/imagine2touch.npz` and replaces the paper's
dataset.

## Settings that are not on the command line

Three settings of the devices cannot be passed to the scripts as options. To change one, edit the
line that creates the device in the command's script under `src/imagine2touch/procedures/scripts/`, for example `ReSkinSensor(port="/dev/ttyUSB0")`.

| Setting | Default | Where |
|---|---|---|
| Serial port of the ReSkin | `/dev/ttyACM0` | `ReSkinSensor(port=...)` |
| Name of the server's node | `/robot_io_ros_server`; the public launch file, without the patch, uses `/panda_ros_server` | `FrankaRobot(server=...)` |
| Position of the sensor in the tool frame the server reports (step 2) | none: the tool frame is the sensor | `FrankaRobot(sensor_in_tcp=...)` |

## Where this differs from the paper's setup

- **The table.** For the paper's recordings, a second camera on a tripod found the table's position
  from two markers, one on the table and one at a known place beside the robot. Here the board from
  step 3 provides it, so no second camera is needed.
- **Waiting for the arm.** The paper's server reported joint velocities; the public one does not.
  Without them, the arm is treated as still once its joint positions stop changing.
