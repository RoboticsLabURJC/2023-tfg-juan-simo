import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import csv

import matplotlib.pyplot as plt
from albumentations import Compose, Affine, ReplayCompose

MAX_ANGULAR = 2.5 # 5 Nürburgring line
MAX_LINEAR = 20 # 12 in some maps be fast
MIN_LINEAR = 3


def load_data(folder):
    name_folder = folder #+ '/' #+ '/Images/'
    list_images = glob.glob(name_folder + '/*.png')
    images = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    name_file = folder + '/data.csv' #'/data.json'
    file = open(name_file, 'r')
    reader = csv.DictReader(file)
    data = []
    for row in reader: # reading all values
        data.append((row['v'], row['w']))
    file.close()
    return images, data

def get_images(list_images, type_image, array_imgs):
    # Read the images
    for name in tqdm(list_images):
        img = cv2.imread(name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[0:240, 0:640]
            # # Not stored cropped:
            # img = img[240:480, 0:640]
            # img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
            img = cv2.resize(img, (int(200), int(66)))
           
        else:
            target_height = int(66)
            target_width = int(target_height * img.shape[1]/img.shape[0])
            img_resized = cv2.resize(img, (target_width, target_height))
            padding_left = int((200 - target_width)/2)
            padding_right = 200 - target_width - padding_left
            img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
        array_imgs.append(img)

    return array_imgs

def parse_json(data, array):
    # Process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array.append((float(v), float(w)))

    return array

def parse_csv(data, array):
    # Process csv
    for v, w in data:
        array.append((float(v), float(w)))
        
    # Find max and min values of w
    max_w = max(array, key=lambda x: x[1])[1]
    min_w = min(array, key=lambda x: x[1])[1]
    print(max_w, min_w)

    return array

def afine_data(array, imgs):
    
    augment = ReplayCompose([
        Affine(p=0.5, rotate=0, translate_percent={'x':(-0.4, 0.4)})
    ])
    
    new_imgs = []
    new_array = []
    for i in tqdm(range(len(imgs))):
        aug = augment(image=imgs[i])
        augmented_img = aug["image"]
        new_imgs.append(augmented_img)
        if aug["replay"]["transforms"][0]["applied"] == True:
            x_transformation_value = aug["replay"]["transforms"][0]["translate_percent"]["x"][1]
            value = aug["replay"]["transforms"][0]["params"]["matrix"].params[0][2]
            new_value = value / 25 * x_transformation_value
            new_array.append((array[i][0], array[i][1] + new_value))
            
            # print("Old: ", array[i], "; New: ", new_array[i])
            # cv2.imshow('Imagen', imgs[i])
            # cv2.imshow('Imagen2', augmented_img)
            # # Esperar a que el usuario presione una tecla para cerrar la ventana
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            # If no transformation applied, keep original values
            new_array.append(array[i])
            
            
    ret_array = array + new_array
    ret_array_imgs = imgs + new_imgs
    return ret_array, ret_array_imgs

def preprocess_data(array, imgs, data_type, afine, norm):
    # Data augmentation
    # Take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip = []
    for i in tqdm(range(len(imgs))):
        # cv2.imshow('Imagen', imgs[i])
        # cv2.imshow('Imagen2', cv2.flip(imgs[i], 1))
        # # Esperar a que el usuario presione una tecla para cerrar la ventana
        # cv2.waitKey(0)
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip.append((array[i][0], -array[i][1]))
    new_array = array + array_flip
    new_array_imgs = imgs + flip_imgs

    # if data_type == 'extreme':
    #     extreme_case_1_img = []
    #     extreme_case_2_img = []
    #     extreme_case_1_array = []
    #     extreme_case_2_array = []

    #     for i in tqdm(range(len(new_array_imgs))):
    #         if abs(new_array[i][1]) > 2:
    #             extreme_case_2_img.append(new_array_imgs[i])
    #             extreme_case_2_array.append(new_array[i])
    #         elif abs(new_array[i][1]) > 1:
    #             extreme_case_1_img.append(new_array_imgs[i])
    #             extreme_case_1_array.append(new_array[i])

    #     new_array += extreme_case_1_array*5 + extreme_case_2_array*10
    #     new_array_imgs += extreme_case_1_img*5 + extreme_case_2_img*10
        
    if afine:
        print('*'*8, "Afinning", '*'*8)
        new_array, new_array_imgs = afine_data(new_array, new_array_imgs)

    if norm:
        print('*'*8, "Normalizing", '*'*8)
        new_array = normalize_annotations(new_array)
        
    return new_array, new_array_imgs

def normalize_annotations(array_annotations):
    array_annotations_v = []
    array_annotations_w = []
    for annotation in array_annotations:
        array_annotations_v.append(annotation[0])
        array_annotations_w.append(annotation[1])
        
    # START NORMALIZE DATA
    array_annotations_v = np.stack(array_annotations_v, axis=0)
    array_annotations_v = array_annotations_v.reshape(-1, 1)

    array_annotations_w = np.stack(array_annotations_w, axis=0)
    array_annotations_w = array_annotations_w.reshape(-1, 1)

    normalized_X = normalize(array_annotations_v, min=MIN_LINEAR, max=MAX_LINEAR)
    normalized_Y = normalize(array_annotations_w, min=-MAX_ANGULAR, max=MAX_ANGULAR)

    normalized_annotations = []
    for i in range(0, len(normalized_X)):
        normalized_annotations.append([normalized_X.item(i), normalized_Y.item(i)])

    return normalized_annotations

def normalize(x, min, max):
    x = np.asarray(x)
    return (x - min) / (max - min)

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
        
