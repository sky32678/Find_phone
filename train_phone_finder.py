import sys
import os
import numpy as np
from PIL import Image
import torch
import random
import os
import pickle
from sklearn.linear_model import LogisticRegression

def crop_image(image, x, y, radius = 0.05):
    x1 = int((x - radius) * image.size[0])
    y1 = int((y - radius) * image.size[1])
    x2 = int((x + radius) * image.size[0])
    y2 = int((y + radius) * image.size[1])
    return image.crop((x1, y1, x2, y2)).resize((40, 40))

def read_text(path):
    lines = []
    train_info = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        temp = list(line.split())
        train_info[temp[0]] = [float(temp[1]), float(temp[2])]
    return train_info

def read_phone_image(path):
    img_list = os.listdir(path)
    return img_list

def random_coor(coor):
    x = random.random()
    y = random.random()
    while (x > coor[0] + 0.1 or x < coor[0] - 0.1) == False:
        x = random.random()
    while (y > coor[1] + 0.1 or y < coor[1] - 0.1) == False:
        y = random.random()
    return x,y

def train(train_info, img_list, path):
    lables = []
    images = []
    for img in img_list:
        # print(img, train_info[img])
        phone_image = Image.open(path+img)

        cropped_image = crop_image(phone_image, train_info[img][0], train_info[img][1])
        images.append((np.array(cropped_image).reshape(-1)).tolist())
        lables.append(1)

        rand_x, rand_y = random_coor(train_info[img])
        cropped_image = crop_image(phone_image, rand_x, rand_y)
        images.append((np.array(cropped_image).reshape(-1)).tolist())
        lables.append(0)

    model = LogisticRegression()
    model.fit(images,lables)
    return model

def main():
    print("TRAINING START")
    label_txt_file_path = "labels.txt"
    image_files_path = sys.argv[1] + "/"
    train_info = read_text(label_txt_file_path)
    img_list = read_phone_image(image_files_path)
    model = train(train_info, img_list, image_files_path)
    print("TRAINING DONE")
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)

if __name__ == "__main__":
    main()
