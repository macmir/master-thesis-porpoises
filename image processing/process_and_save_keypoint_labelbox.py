import json
import cv2 as cv
import os
import numpy as np
import pandas as pd

# images
path = '/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/labeled_train/output'
files = os.listdir(path)
files.sort()

# file with annotations
labels = pd.read_json('/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/labeled_train/results.json')

save_path = '/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/labeled_train_processed'
name_increment = 0
annotations_file = {}

for i in range (len(labels.columns)):

    img = cv.imread(path + '/' + labels.iloc[:, i].name)

    label_0_x = int(labels.iloc[0, i][0])
    label_0_y = int(labels.iloc[0, i][1])

    label_1_x = int(labels.iloc[1, i][0])
    label_1_y = int(labels.iloc[1, i][1])

    # cv.circle(img, (label_0_x, label_0_y) , radius=1, thickness=2, color=(0, 255, 0))
    # cv.circle(img, (label_1_x, label_1_y) , radius=1, thickness=2, color=(0, 0, 255))
    # cv.imshow('original', img)
    # cv.waitKey(0)

    if img.shape[0] > img.shape[1]: # taller

        ratio = 128 / img.shape[0]
        img = cv.resize(img, (0, 0), fx=ratio, fy=ratio)
        blackbars = (128 - img.shape[1]) // 2
        square = np.zeros((128, 128, 3), dtype=np.uint8)
        square[:, blackbars:img.shape[1] + blackbars] = img

        label_0_x_ = int(label_0_x * ratio) + blackbars
        label_0_y_ = int(label_0_y * ratio) 
        label_1_x_ = int(label_1_x * ratio) + blackbars
        label_1_y_ = int(label_1_y * ratio) 

        # cv.circle(square, (label_0_x_, label_0_y_) , radius=1, thickness=1, color=(0, 255, 0))
        # cv.circle(square, (label_1_x_, label_1_y_) , radius=1, thickness=1, color=(0, 0, 255))
        # cv.imshow('processed - taller', square)
        # print(square.shape  )
        # cv.waitKey(0)

        cv.imwrite(save_path + '/' + 'labeled_train_' + str(name_increment) + '.png', square)
        annotations_file['labeled_train_' + str(name_increment) + '.png'] = {
                        'tongue': [label_0_x_, label_0_y_],
                        'tail': [label_1_x_, label_1_y_]
                    }
        name_increment += 1 

    if img.shape[0] < img.shape[1]: # wider

        ratio = 128 / img.shape[1]
        img = cv.resize(img, (0, 0), fx=ratio, fy=ratio)
        blackbars = (128 - img.shape[0]) // 2
        square = np.zeros((128, 128, 3), dtype=np.uint8)
        square[blackbars:img.shape[0] + blackbars:] = img

        label_0_x_ = int(label_0_x * ratio) 
        label_0_y_ = int(label_0_y * ratio) + blackbars
        label_1_x_ = int(label_1_x * ratio) 
        label_1_y_ = int(label_1_y * ratio) + blackbars

        # cv.circle(square, (label_0_x_, label_0_y_) , radius=1, thickness=1, color=(0, 255, 0))
        # cv.circle(square, (label_1_x_, label_1_y_) , radius=1, thickness=1, color=(0, 0, 255))
        # cv.imshow('processed - wider', square)
        # print(square.shape  )
        # cv.waitKey(0)

        cv.imwrite(save_path + '/' + 'labeled_train_' + str(name_increment) + '.png', square)
        annotations_file['labeled_train_' + str(name_increment) + '.png'] = {
                        'tongue': [label_0_x_, label_0_y_],             
                        'tail': [label_1_x_, label_1_y_]
                    }
        name_increment += 1 

save_file = open('json_with_keypoints_labeled_train.json', 'w')
json.dump(annotations_file, save_file)
save_file.close()