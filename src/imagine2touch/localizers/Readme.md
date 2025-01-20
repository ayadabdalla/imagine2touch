# Wrist cam localizer
- place a physical apriltag grid on the world frame, given that you approx. measured the robot_world_transform.
- configure the script using ```wrist.yaml``` in ```config``` directory given that you have the physical apriltag ```json``` file.
- run ```wrist_cam_apriltag_localizer.py```
- run ``` wrist_cam_calibration_optimization.py``` to calculate the wrist_camera_tcp_transform using the data collected by the localizer.