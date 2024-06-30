# Implementation of a neural network algorithm for keypoint detection of porpoises and research on the synthetic data usage

This repository contains code for master thesis regarding keypoint detection of porpoises. The solution is based on repository: https://github.com/tlpss/keypoint-detection . The algorithm was modified, so that the two keypoints of a porpoise (snout and tail) are detected on input images. The dataset creation and image processing scripts are in **image processing** directory. Training process is launched from train.py script in **tasks** directory. Exemplary training command:

```bash
python3 -m keypoint_detection.tasks.train  \
--keypoint_channel_configuration  "tail: tongue" \
--json_dataset_path <train_dataset_path> \
--json_test_dataset_path <test_dataset_path> \
--batch_size  2 \
--wandb_project "keypoint-detector-integration-test" \
--max_epochs 75 \
--early_stopping_relative_threshold -1.0 \
--log_every_n_steps 1 \
--accelerator="gpu" \
--devices 1 \
--precision 16 \
--augment_train \
--heatmap_sigma 1 \
--maximal_gt_keypoint_pixel_distances "3 5 7 9" \
--max_keypoints 1 \
--validation_split_ratio 0.15 \
--backbone_type "ConvNeXtUnet"
```
