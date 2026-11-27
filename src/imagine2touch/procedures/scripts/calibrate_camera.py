"""
Find where the wrist camera sits on the sensor and where the table is (needs the robot).

    python -m imagine2touch.procedures.scripts.calibrate_camera --print-board board.pdf
    python -m imagine2touch.procedures.scripts.calibrate_camera --board-centre 0.45 0.0 0.0 --out calibration
    python -m imagine2touch.procedures.scripts.calibrate_camera --check

Print the board at its real size, without scaling it to the page, and lay it flat where the objects
will stand.  Its tags are then 30 mm wide; if yours came out another width, pass it as --tag [m].
--board-centre is roughly where the board's middle is, in the frame of the robot's base [m].  The
robot then looks at the board from 13 directions.  Two files are written:
`camera.npy`, the camera in the frame of the sensor, and `world.npy`, the board in the frame of the
robot's base.  They are the --camera and --world of the other scripts: the board's frame is the
world frame, with its origin at the lower left tag and its z axis pointing up from the table.

--check is a quick test without hardware: the 13 moves and the calculation are run on a robot and a
board that exist only in memory.  No tag is detected, so it says nothing about a real camera.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from imagine2touch.datasets import VisuoTactileDataset
from imagine2touch.devices.april_tag_board import AprilTagBoard
from imagine2touch.procedures import CameraCalibrator
from imagine2touch.simulation import SimulatedBoard, SimulatedCamera, SimulatedRobot, SimulatedScene

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-board", metavar="FILE", help="write the board to print, as a .pdf or an image, and stop")
    ap.add_argument("--tag", type=float, default=0.03, help="width of one tag [m], on the sheet to print or as printed")
    ap.add_argument("--board-centre", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--expected", help="a rough guess of the camera in the sensor's frame, 4 x 4 .npy (default: the paper's mount)")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--check", action="store_true", help="test the moves and the calculation without hardware")
    ap.add_argument("--object", default="star", help="with --check: the object whose table the board lies on")
    args = ap.parse_args()
    expected = np.load(args.expected) if args.expected else np.round(SimulatedCamera.MOUNT, 2)
    if args.print_board:
        AprilTagBoard(tag=args.tag).save(args.print_board)
        raise SystemExit(f"wrote {args.print_board}: print it at its real size, without scaling to the page; "
                         f"a tag is then {1000 * args.tag:.0f} mm wide")
    if args.check:
        scene = SimulatedScene.of_session(VisuoTactileDataset(), args.object)
        robot = SimulatedRobot(scene)
        camera = SimulatedCamera(robot, width=64, height=48, focal=120.0)      # the simulated board reads no image
        board, centre, sleep = SimulatedBoard(robot, camera.camera_in_sensor, scene.base_from_world), scene.to_base([0, 0, 0]), lambda seconds: None
    else:
        from imagine2touch.devices.franka_robot import FrankaRobot
        from imagine2touch.devices.realsense_camera import RealSenseCamera
        if args.board_centre is None:
            ap.error("--board-centre is needed with the robot")
        robot, camera, board, centre, sleep = FrankaRobot(), RealSenseCamera(), AprilTagBoard(tag=args.tag), np.array(args.board_centre), time.sleep
    calibration = CameraCalibrator(robot, camera, board, expected, sleep=sleep).calibrate(centre)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calibration.save(out / "camera.npy", out / "world.npy")
    print(f"{len(calibration.spread)} views; they place the board within {1000 * calibration.spread.mean():.1f} mm and "
          f"{np.degrees(calibration.turn.mean()):.2f} degrees of each other on average")
    print(f"camera in the sensor's frame [m]:\n{np.round(calibration.camera_in_sensor.matrix, 4)}")
    print(f"wrote {out / 'camera.npy'} and {out / 'world.npy'}")
