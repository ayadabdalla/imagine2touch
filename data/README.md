# Data

Two files go here, both from the
[datasets folder](https://drive.google.com/drive/folders/1R-VS56TSSfveVk3LtNUCnogSlWpfhxf7?usp=sharing):

- `imagine2touch.npz` (8 MB): the dataset of the paper, 1,630 touches of six objects (`flower`,
  `hexagon`, `parallelogram`, `star`, `square`, `cutter`).
- `shape_classification.npz`: 1,280 presses of five stamps into the mounted sensor.

`imagine2touch.npz` can also be rebuilt from the
[raw recordings](https://drive.google.com/drive/folders/1dU-ReQk-5dL5UJNuePnRpuHHKqbfHsXU?usp=drive_link):

```bash
python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data
```

`scans/hanger.npz` is part of the repository: the scan of the hanger, the one candidate of the paper's
recognition experiment that was never touched for the dataset. It is made from the point cloud of the
original repository (`reskin/task/data/hanger/hanger_combined.pcd`) by

```bash
python -m imagine2touch.datasets.recordings.scan_converter --source PATH/TO/hanger_combined.pcd --out data/scans/hanger.npz
```

What the files hold is described in `src/imagine2touch/datasets/visuo_tactile_dataset.py` and
`shape_classification_dataset.py`; the layout of the recordings in `src/imagine2touch/datasets/recordings/recording.py`.
