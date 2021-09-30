import sys
import os
from PIL import Image
import torch
import pickle
from sklearn.linear_model import LogisticRegression
from train_phone_finder import read_phone_image, crop_image, read_text
import numpy as np

import matplotlib.pyplot as plt

def test(model, img_list, path, train_info, radius = 0.05, division = 2.5):
    count = 0
    ans = 0
    num_img = len(img_list)
    for img in img_list:
        phone_image = Image.open(path+img)
        w = phone_image.size[0]
        d = phone_image.size[1]
        de_norm_w = phone_image.size[0] * radius
        de_norm_d = phone_image.size[1] * radius

        curr_x = radius
        curr_y = radius

        cropped_images = []
        coor = []

        for i in range(int(w/de_norm_w*division)):
            for j in range(int(w/de_norm_w*division)):
                cropped_image = crop_image(phone_image, curr_x, curr_y)
                cropped_images.append((np.array(cropped_image).reshape(-1)).tolist())
                coor.append([curr_x, curr_y])
                curr_x += (radius/division)
                if curr_x > 1 - radius:
                    break
            curr_x = radius
            curr_y += (radius/division)
            if curr_y > 1 - radius:
                break

        pridict = model.predict_proba(cropped_images)
        sol = np.argmax(pridict[:, 1])
        expected_x, expected_y = coor[sol][0] , coor[sol][1]
        actual_x , actual_y = train_info[img][0], train_info[img][1]

        if ((actual_x - expected_x)**2 + (actual_y - expected_y)**2)**(0.5) <= 0.05:
            ans += 1
        count += 1
        print(count, "/", num_img, int(count/num_img*100),"%")
    return ans/count * 100

def main():
    print("TRAINING SET TEST")
    label_txt_file_path = "labels.txt"
    image_files_path = "find_phone/"
    train_info = read_text(label_txt_file_path)
    img_list = read_phone_image(image_files_path)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    ans = test(model, img_list, image_files_path, train_info)
    print("Accuracy",ans,"%")

if __name__ == "__main__":
    main()
