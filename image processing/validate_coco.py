# check the final .json file in coco format, see if the keypoints are in correct places

import json
import cv2 as cv


data_path = '/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/real_training/'

with open('real_train_250.json') as f:
    data = json.load(f)
a = 0
if 'images' in data:
    image_keypoints = []
    for image in data['images']:
        image_id = image['id']
        image_name = image['file_name']
        keypoints = []
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                keypoints.append(annotation['keypoints'])
        image_keypoints.append({'image_name': image_name, 'keypoints': keypoints})

    for item in image_keypoints:
        a+=1
        tongue = item['keypoints'][0][0:2]
        tail = item['keypoints'][0][3:5]
        img = cv.imread(data_path + item['image_name'])
        cv.circle(img, (tongue[0], tongue[1]), 1, (0, 255, 0), 1)
        cv.circle(img, (tail[0], tail[1]), 1, (0, 255, 255), 1)

        cv.imshow('', img)
        print(item['image_name'])
        cv.waitKey(0)

else:
    print("'images' key not found in the JSON data.")
print(a                )