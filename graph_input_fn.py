import cv2
import os 
import numpy as np
import os.path

from tensorflow.keras.preprocessing.image import img_to_array

calib_img_path = "./dataset/leftImg8bit_trainvaltest/leftImg8bit/train"

calib_batch_size = 1

def calib_input(iter):
    
    paths = []
    for (path, dirname, files) in sorted(os.walk(calib_img_path)):
        for filename in sorted(files):
            if filename.endswith(('.jpg', '.png')):
                paths.append(os.path.join(path, filename))

    images = []
    for index in range(0, calib_batch_size):
        #print("Path: " + paths[(iter * calib_batch_size + index) % len(paths)])
        img = cv2.imread(paths[(iter * calib_batch_size + index) % len(paths)], 1)
        try:
            img = cv2.resize(img,(512, 256))
            images.append(img)
#            cv2.imshow('img',img)
        except cv2.error as e:
            print('Invalid frame!')
            print(e)

    return {"input_1": images}

def main():
    calib_input()

if __name__ =="__main__":
    main()
