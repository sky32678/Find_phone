import sys
import os
from PIL import Image
import torch
import pickle
from sklearn.linear_model import LogisticRegression
from train_phone_finder import read_phone_image, crop_image
import numpy as np

def find_phone_coor(path, radius = 0.05, division = 4):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    phone_image = Image.open(path)
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
    sol = np.argmax(pridict[:,1])

    return print(coor[sol][0], coor[sol][1])


def main():
    find_phone_coor(sys.argv[1])

if __name__ == "__main__":
    main()
