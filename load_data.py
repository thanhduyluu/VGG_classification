
import pickle
import numpy as np
import os
import cv2 as cv


label = os.listdir("dataset_image")
label=label[1:]
dataset=[]

for image_label in label:

    images = os.listdir("dataset_image/"+image_label)

    for image in images:
        img = cv.imread("dataset_image/"+image_label+"/"+image,0)
        img = cv.resize(img, (244,244))
        x = np.array(img, dtype="uint8")
        x = x.reshape((244, 244, 1))
        dataset.append((x, image_label))


X=[]
Y=[]

for input, image_label in dataset:
    X.append(input)

    Y.append(label.index(image_label))


X=np.array(X, dtype="uint8" )
Y=np.array(Y, dtype="uint8")



X_train,y_train  = X,Y


data_set=(X_train,y_train)



save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()
