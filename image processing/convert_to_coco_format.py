# Export final .json file in coco format for training

import json


with open('json_with_keypoints.json') as f:
# with open('data/labelbox/output.json') as f:
    data = json.load(f)
coco_data = {
   "info": {
        "description": "test dataset for keypoint detection",
        "url": "put.poznan.pl",
        "version": "0.0.1",
        "year": "2024",
        "contributor": "Maciek Mirecki",
        "date_created": "2024/02/21"

    },
    "licenses": [
        {
            "url": "hhttps://creativecommons.org/licenses/by/2.0/",
            "id": 1,
            "name": "Attribution Generic License"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "box",
            "keypoints": [
                "tongue",
                "tail"
            ],
            "skeleton": [],
            "supercategory": "box"
        }
    ],
    "images": [],
    "annotations": []

}

for image_name, keypoints in data.items():
    image_id = len(coco_data['images']) + 1

    coco_data['images'].append({
        "id": image_id,
        "license": 1,
        "file_name": 'images_new/' + image_name + '.png',
        "width": 128, 
        "height": 128, 
    })

    coco_data['annotations'].append({
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": [
            keypoints['tongue'][0], keypoints['tongue'][1],
            2,
            keypoints['tail'][0], keypoints['tail'][1],
            2,
        ],
        "num_keypoints": 2,

    })

with open('keypoints_coco.json', 'w') as f:
    json.dump(coco_data, f, indent=4)