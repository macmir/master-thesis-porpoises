from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import time

device = 'cuda'
checkpoint_path = '/home/maciek/Desktop/model_convnext.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
backbone = BackboneFactory.create_backbone(**checkpoint["hyper_parameters"])
model = KeypointDetector.load_from_checkpoint(checkpoint_path, backbone=backbone)
model.to(device)

def run_inference(model: KeypointDetector, image, confidence_threshold: float = 0.1) -> Image:
    model.eval()
    tensored_image = torch.from_numpy(np.array(image)).float()
    tensored_image = tensored_image / 255.0
    tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)

    with torch.no_grad():
        heatmaps = model(tensored_image.to(device))

    keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=confidence_threshold)
    image_keypoints = keypoints[0]
    tail = image_keypoints[0][0]
    tongue = image_keypoints[1][0]

    draw = ImageDraw.Draw(image)
    radius=3

    draw.ellipse((tail[0] - radius, tail[1] - radius, tail[0] + radius, tail[1] + radius), fill="red")
    draw.ellipse((tongue[0] - radius, tongue[1] - radius, tongue[0] + radius, tongue[1] + radius), fill="green")
    image.save("res.png")

    return image


if __name__ == "__main__":
    image_path = "/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/labeled_train_processed/labeled_train_87.png"
    image_files = os.listdir("/home/maciek/Desktop/dataset")
    folder_path = "/home/maciek/Desktop/dataset"
    images = [] 
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        images.append(image)

    start_time = time.time()

    for img in images:
        run_inference(model, image)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total inferencing time: {round(total_time, 3)}')
    print(f'Inferencing time per image: {round(total_time/10, 3)}')

