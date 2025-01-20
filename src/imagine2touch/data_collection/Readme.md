# Readme

# Experiments

To start a new data collection experiment:

- Edit the configurations in the ```collection.yaml``` file in the ```cfg``` directory.
    
# Post processing

- `poses_processing.py` processes the stored proprioception information from experiments.  It is required to run on your target test data before running `reconstruct_pcd.py` script in the ```task``` directory.

- Before running it make sure to edit the ```poses.yaml``` file in the ```cfg``` directory

# PCDs filters

- For segmentation of unwanted extremeties run ```process_pcd_extremeties.py``` and interact with the script
- For filtering sparse chunks of points run ```quick_pcd_filter.py```