# check keypoint positions after processing synthetic images

import pandas as pd
import cv2 as cv

annotations_file = pd.read_json('json_with_keypoints_labeled_train.json')
print(len(annotations_file.columns))
print(annotations_file.head())
count = 0
for i in range (len(annotations_file.columns)):
    count+=1 
    print(annotations_file.iloc[:, i].name)
    print(annotations_file.iloc[0, i]) # tongue
    print(annotations_file.iloc[1, i]) # tail

    tongue = annotations_file.iloc[0, i]
    tail = annotations_file.iloc[1, i]

    img = cv.imread('/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/labeled_train_processed/'+str(annotations_file.iloc[:, i].name))
    cv.circle(img, (int(tongue[0]), int(tongue[1])) , radius=1, thickness=2, color=(0, 255, 0))
    cv.circle(img, (int(tail[0]), int(tail[1])) , radius=1, thickness=2, color=(0, 0, 255))
    cv.imshow('', img)
    cv.waitKey(0)

print(count)