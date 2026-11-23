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

A third file is only needed for `configs/v6.yaml`: `imagine2touch_v6.npz`,
the six sessions recorded in September 2023, after the paper's. They are part of the raw recordings;
build the file from them:

```bash
python -m imagine2touch.datasets.recordings.recording_reducer --source PATH/TO/Imagine2touch_data \
    --sessions cylinder_v_6 flower_v_6 hexagon_v_6 parallelogram_v_6 square_v_6 star_v_6 \
    --out data/imagine2touch_v6.npz
```

What the files hold is described in `src/imagine2touch/datasets/visuo_tactile_dataset.py` and
`shape_classification_dataset.py`; the layout of the recordings in `src/imagine2touch/datasets/recordings/recording.py`.
