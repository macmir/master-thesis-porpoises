# Processing synthetic images, saving image name and keypoints to .json file

import json
import cv2 as cv
import os
import numpy as np


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        return {}

def append_to_json(filename, data):
    existing_data = load_json(filename)
    existing_data.update(data)
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)


json_file = load_json('empty_json22.json')
save_path = '/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/test_process/'
folders = ['data/synthetic/solo_1/sequence.0', 'data/synthetic/solo_2/sequence.0', 'data/synthetic/solo_3/sequence.0', 'data/synthetic/solo_4/sequence.0']
# folders = ['data/synthetic/solo_1/sequence.0']
name_increment = 0
for folder in folders:
    data_path = folder

    files = os.listdir(data_path)
    files.sort()

    bounding_box_extension = 10
    scaling_factor = 0.8
    k = 0
   
    annotations_file = {}

    for i in range(0, len(files)-1, 2):
        img_path = data_path + '/' + files[i]
        json_path = data_path + '/' + files[i + 1]
        data = json.load(open(json_path))

        bounding_boxes = []
        keypoints = []

        for annotation in data['captures'][0]['annotations']:
            if annotation['@type'] == 'type.unity.com/unity.solo.BoundingBox2DAnnotation':
                bounding_boxes.extend(annotation['values'])
            elif annotation['@type'] == 'type.unity.com/unity.solo.KeypointAnnotation':
                keypoints.extend(annotation['values'])

        for i in range(len(bounding_boxes)):
            img = cv.imread(img_path, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_LINEAR)
            x = int(bounding_boxes[i]['origin'][0]) - bounding_box_extension - 3
            y = int(bounding_boxes[i]['origin'][1]) - bounding_box_extension - 3
            w = int(bounding_boxes[i]['dimension'][0]) + bounding_box_extension * 2
            h = int(bounding_boxes[i]['dimension'][1]) + bounding_box_extension * 2

            keypoint = keypoints[i]
            keypoint_info = [(kp['index'], kp['location']) for kp in keypoint['keypoints']]

            label0_x = int(keypoint_info[0][1][0])
            label0_y = int(keypoint_info[0][1][1])
            label1_x = int(keypoint_info[1][1][0])
            label1_y = int(keypoint_info[1][1][1])

            # Scaled bounding box dimensions and keypoints
            x = int(x * scaling_factor)
            y = int(y * scaling_factor)
            w = int(w * scaling_factor)
            h = int(h * scaling_factor)
            label0_x = int(label0_x * scaling_factor)
            label0_y = int(label0_y * scaling_factor)
            label1_x = int(label1_x * scaling_factor)
            label1_y = int(label1_y * scaling_factor)

            # Keupoints with reference to boundingbox
            label0_x_bb = label0_x - x
            label0_y_bb = label0_y - y
            label1_x_bb = label1_x - x
            label1_y_bb = label1_y - y

            if 0 < label0_x_bb < w and 0 <label0_y_bb < h and 0 < label1_x_bb < w and 0 < label1_y_bb < h and w != 0 and h != 0 and x > 0 and y > 0:

                # cv.circle(img, (label0_x, label0_y), 1, (0, 255, 255))
                # cv.circle(img, (label1_x, label1_y), 1, (0, 255, 255))
                bbox = img[y:y + h, x:x + w]
                bbox_org = img[y:y + h, x:x + w]

                if bbox.shape[0] > bbox.shape[1]:  # Taller
                    ratio = 128 / bbox.shape[0]
                    # print(ratio)
                    # print(bbox.shape)
                    # print(w, h, x, y)
                    bbox = cv.resize(bbox, (0, 0), fx=ratio, fy=ratio)
                    blackbars = (128 - bbox.shape[1]) // 2
                    square = np.zeros((128, 128, 3), dtype=np.uint8)
                    square[:, blackbars:bbox.shape[1] + blackbars] = bbox

                    label0_x_bb = int(label0_x_bb * ratio)
                    label0_y_bb = int(label0_y_bb * ratio)
                    label1_x_bb = int(label1_x_bb * ratio)
                    label1_y_bb = int(label1_y_bb * ratio)
                    cv.imwrite(save_path +  str(name_increment) + '.png', square)

                    annotations_file[str(name_increment) + '.png'] = {
                        'tongue': [label1_x_bb + blackbars, label1_y_bb],
                        'tail': [label0_x_bb + blackbars, label0_y_bb]
                    }
                    image_data = {
                        str(name_increment): {"tongue": [label1_x_bb + blackbars, label1_y_bb], "tail": [label0_x_bb + blackbars, label0_y_bb]},
                    }
                    append_to_json("empty_data.json", image_data)

                    print(f'File saved as:', str(name_increment) + '.png\n')

                    # Save file
                    # save_file = open('/home/maciek/Documents/Studia/Magisterka/master-thesis/dataset2/annotations/' + str(name_increment) + '.json', 'w')
                    # json.dump(kp, save_file)
                    # save_file.close()
                    name_increment += 1

                    # cv.circle(bbox, (label0_x_bb, label0_y_bb), 1, (0, 0, 255))
                    # cv.circle(bbox, (label1_x_bb, label1_y_bb), 1, (0, 0, 255))
                    
                    # cv.circle(square, (label0_x_bb + blackbars, label0_y_bb), 1, (0, 255, 0))
                    # cv.circle(square, (label1_x_bb + blackbars, label1_y_bb), 1, (0, 255, 0))
                    # cv.imshow('square 128x128', square)
                    # cv.imshow('original bbox', bbox_org)
                    # cv.imshow('Resized bbox', bbox)
                    # cv.waitKey(0)
                    # time.sleep(0.25)

                if bbox.shape[0] < bbox.shape[1]:  # Wider
                    ratio = 128 / bbox.shape[1]
                    bbox = cv.resize(bbox, (0, 0), fx=ratio, fy=ratio)
                    blackbars = (128 - bbox.shape[0]) // 2
                    square = np.zeros((128, 128, 3), dtype=np.uint8)
                    square[blackbars:bbox.shape[0] + blackbars:] = bbox

                    label0_x_bb = int(label0_x_bb * ratio)
                    label0_y_bb = int(label0_y_bb * ratio)
                    label1_x_bb = int(label1_x_bb * ratio)
                    label1_y_bb = int(label1_y_bb * ratio)

                    if (label0_y_bb + blackbars) > 128 or (label0_y_bb + blackbars) > 128:
                        break
                    if label0_x_bb > 128 or label1_x_bb > 128:
                        break
                    cv.imwrite(save_path +  str(name_increment) + '.png', square)
                    annotations_file[str(name_increment) + '.png'] = {
                        'tongue': [label1_x_bb, label1_y_bb + blackbars],
                        'tail': [label0_x_bb, label0_y_bb + blackbars]
                    }
                    image_data = {
                        str(name_increment): {"tongue": [label1_x_bb, label1_y_bb + blackbars], "tail": [label0_x_bb, label0_y_bb + blackbars]},
                    }
                    append_to_json("empty_data.json", image_data)                    

                    print(f'File saved as:', str(name_increment) + '.png\n')
                    
                    name_increment += 1

            else:
                k += 1
save_file = open('json_with_keypoints22.json', 'w')
json.dump(annotations_file, save_file)
save_file.close()

print('Number of wrong keypoints coordinates: ', k)
