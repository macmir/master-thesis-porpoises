import wandb
import torch
from keypoint_detection.models.detector import KeypointDetector
from torch.autograd import Variable


# Initialize a run
run = wandb.init(project="keypoint-detector-integration-test", entity="mac-mir")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="mac-mir/keypoint-detector-integration-test/model-nw0gpq1e:v0")

model = KeypointDetector(heatmap_sigma=3, maximal_gt_keypoint_pixel_distances=[3, 5, 7, 9],
                         minimal_keypoint_extraction_pixel_distance=1, learning_rate=0.0003, backbone='ConvNeXtUnet',
                         keypoint_channel_configuration=['tail', 'tongue'], ap_epoch_start=1, ap_epoch_freq=2,
                         lr_scheduler_relative_threshold=0, max_keypoints=1)
dummy_input = Variable(torch.randn(22, 3, 128, 128)).cuda()

torch.onnx.export(model,dummy_input , downloaded_model_path, verbose=False)
