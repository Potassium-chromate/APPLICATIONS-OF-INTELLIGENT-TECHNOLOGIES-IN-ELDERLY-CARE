# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:54:47 2023

@author: Eason
"""


import os
from PIL import Image
import numpy as np
# Define your CNN model here (this is just an example)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from scipy import ndimage
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50,InceptionV3, VGG19, VGG16, MobileNetV2 
from tensorflow.keras.optimizers import Adam
import random

# Define the path to the folder containing images
path = 'C:/Users/88696/Desktop/四上課程/智慧科技於高齡照護的應用/project'

def load(path,target_size,arg = 'yes'):
    ret_data = []
    ret_label = []
    ret_name = []
    count = 0

    category = 0
    file_list = os.listdir(path)
    # Filter out non-image files (if any)
    
    for f in file_list: #f is image name is the folder
        folder_path = os.path.join(path, f)
        img_list = os.listdir(folder_path)
        if(len(img_list) >= 4):
            category = category + 1 

            
    
    #img_files = [f for f in file_list if os.path.splitext(f)[1].lower() in img_extensions]
    for f in file_list: #f is image name is the folder
        folder_path = os.path.join(path, f)
        img_list = os.listdir(folder_path)
        if(len(img_list)<4):
            continue; 
        for i in img_list:
           
            img_path = os.path.join(folder_path, i)
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize(target_size)
        
            # Original image
            img_array = np.array(img)
            img_array = img_array/255
            label = np.zeros(category)
            label[count] = 1 #Create one-hot encoding label
            ret_data.append(img_array)
            ret_label.append(label)
            ret_name.append(i)
        
        
        count += 1
  


    ret_data , ret_label , ret_name = np.array(ret_data),np.array(ret_label),np.array(ret_name)
    return ret_data,ret_label,ret_name

def arg(data, label, angles=[30, 60, -30, -60]):
    ret_data = []
    ret_label = []
    
    for i in range(len(data)):
        image = data[i]
        ret_data.append(image)  # Original image
        ret_label.append(label[i])
        
        # Horizontal and vertical flips
        image_flip_h = np.fliplr(image)
        image_flip_v = np.flipud(image)
        ret_data.extend([image_flip_h, image_flip_v])
        ret_label.extend([label[i], label[i]])
        
        # Rotation
        for angle in angles:
            # Rotate image by 'angle' degrees
            image_rotated = ndimage.rotate(image, angle, reshape=False)
            ret_data.append(image_rotated)
            ret_label.append(label[i])
    
    ret_data = np.array(ret_data)
    ret_label = np.array(ret_label)
    
    
    return ret_data, ret_label
        


ret_data,ret_label,ret_name = load(path,(256,256),arg = 'yes')
# Indices to extract and remove


ret_data, ret_label = arg(ret_data, ret_label)
indices = np.arange(len(ret_data))
np.random.shuffle(indices)
ret_data = ret_data[indices]
ret_label = ret_label[indices]

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

random_numbers = random.sample(range(0, np.shape(ret_data)[0]), 30)
val_data = ret_data[random_numbers]
val_label = ret_label[random_numbers]

train_data = np.delete(ret_data,random_numbers, axis=0)
train_label = np.delete(ret_label,random_numbers, axis=0)


random_numbers = random.sample(range(0, np.shape(train_data)[0]), 30)
temp_data = train_data[random_numbers]
val_data = np.concatenate((val_data, temp_data), axis=0)
temp_label = train_label[random_numbers]
val_label = np.concatenate((val_label, temp_label), axis=0)

base_model = InceptionV3(include_top= False, input_shape = (256, 256, 3), weights= 'imagenet')
for layer in base_model.layers[0:len(base_model.layers)-50]:  
       layer.trainable = False
       
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(np.shape(ret_label)[1], activation='softmax')) # Use 'softmax' for categorical labels

adam_optimizer = Adam(learning_rate=0.001)
# Compile the model
model.compile(loss='categorical_crossentropy', # Use 'categorical_crossentropy' for categorical labels
              optimizer = adam_optimizer,
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_data,train_label, epochs=40, validation_data = (val_data, val_label),callbacks=[reduce_lr])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

