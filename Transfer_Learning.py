#!/usr/bin/env python
# coding: utf-8

print("Step1 started")
import glob
import numpy as np
import os
import shutil
from keras.models import Model
import keras
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns 
np.random.seed(42)
print("Step1 completed")

print("Step2 started")
w, h = 3, 6;
acc_data = [[0 for x in range(w)] for y in range(h)]
print("Step2 completed")


print("Step3 started")
def user_input():
    try:        
        index1 = int(input("Enter desired Image classification methodology from below \n 1.Transfer Learning by Feature Extraction 		\n 2.Transfer Learning by Feature Extraction and Image Augmentation \n 3.Transfer Learning with Fine-tuning and Image 		Augmentation \n"))
        if(index1 not in [1,2,3]):
            print("\033[1;30;41m 'Invalid input : Please choose 1 or 2 or 3 only.'") 
            print("\033[1;30;0m Please check")
            user_input()  
        return index1
    except:
        print("\033[1;30;41m 'Invalid input : Please choose integers 1 or 2 or 3 only.'") 
        print("\033[1;30;0m Please check")
        user_input()
index = user_input()
print("Step3 completed")


print("Step4 started")
def user_input1():
    try:        
        index2 = int(input("Enter the desired Neural Network from below \n 1.VGG16 \t 2.VGG19 \t 3.ResNet50 \t 4.InceptionV3 \t 	5.Xception \t 6.DenseNet\n"))
        if(index2 not in [1,2,3,4,5,6]):
            print("\033[1;30;41m 'Invalid input : Please choose from 1 to 6 only.'") 
            print("\033[1;30;0m Please check")
            user_input1() 
        return index2
    except:
        print("\033[1;30;41m 'Invalid input : Please choose integers from 1 to 6 only.'")
        print("\033[1;30;0m Please check")
        user_input1()
network = user_input1()
print("Step4 completed")



print("Step5 started")
#The data file is arranged as file->3 inner files(test/train/val)->In each(test/train/val) file there are 2 innermost files(NORMAL/PNEUMONIA) 
def createLists(folder, folderName, list1, list2):
    for innerFolder in os.listdir(folder + folderName):
        if innerFolder in ['NORMAL']:
            for image_filename in os.listdir(folder + folderName + '/' + innerFolder):
                img_path = folder + folderName + '/' + innerFolder + '/' + image_filename
                list1.append(img_path) 
        if innerFolder in ['PNEUMONIA']:
            for image_filename in os.listdir(folder + folderName + '/' + innerFolder):
                img_path = folder + folderName + '/' + innerFolder + '/' + image_filename
                list2.append(img_path)
print("Step5 completed")                


print("Step6 started")
#function to create lists for all the three categories of data-test, train and val
x_train=[]
y_train=[]
x_test=[]
y_test=[]
x_val=[]
y_val=[]
def get_files(folder):    
    for folderName in os.listdir(folder):        
        if folderName in ['test']:            
            createLists(folder, folderName, x_test, y_test)
        if folderName in ['train']:
            createLists(folder, folderName, x_train, y_train)
        if folderName in ['val']:
            createLists(folder, folderName, x_val, y_val)                 
    
get_files('chest_xray_fulldataset/')
print(type(x_train))
print(len(x_test))
print(len(y_test))
print(len(x_train))
print(len(y_train))
print(len(x_val))
print(len(y_val))
print("Step6 completed")



print("Step7 started")
x_train = x_train[:1200]
y_train = y_train[:3800]
x_test = x_test[:300]
y_test = y_test[:390]
x_val = x_val[:80]
y_val = y_val[:80]
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_val = np.asarray(x_val)
y_val = np.asarray(x_val)
print(type(x_train))
print(len(x_test))
print(len(y_test))
print(len(x_train))
print(len(y_train))
print(len(x_val))
print(len(y_val))
print("Step7 completed")


print("Step8 started")
# data to plot
def plot(arr1, arr2, typ):
    index = np.arange(1)
    bar_width = 0.2
    opacity = 0.75

    rects1 = plt.bar(index, len(arr1), bar_width,
    alpha=opacity,
    color='g',
    )

    rects2 = plt.bar(index + bar_width, len(arr2), bar_width,
    alpha=opacity,
    color='orange',
    )

    plt.xlabel(typ, size=12)
    plt.ylabel('Count', size=12)               

# create plot
plt.suptitle('Datasets lookup', size=16, y=1.06)
plt.subplot(1,3,1)
plot(x_train, y_train,'Train')
plt.subplot(1,3,2)
plot(x_test, y_test, 'Test')
plt.subplot(1,3,3)
plot(x_val, y_val, 'Val')
plt.figlegend(['NORMAL','PNEUMONIA'], loc='upper left', ncol=1)
plt.tight_layout()
print("Step8 completed")


print("Step9 started")
print("Start")
if os.path.isdir('chest_xray_fulldataset/training_data'):
    shutil.rmtree('chest_xray_fulldataset/training_data')
if os.path.isdir('chest_xray_fulldataset/test_data'):
    shutil.rmtree('chest_xray_fulldataset/test_data')
if os.path.isdir('chest_xray_fulldataset/validation_data'):
    shutil.rmtree('chest_xray_fulldataset/validation_data')
print("Step9 completed")


print("Step10 started")
train_files = np.concatenate([x_train, y_train])
validate_files = np.concatenate([x_val, y_val])
test_files = np.concatenate([x_test, y_test])

os.mkdir('chest_xray_fulldataset/training_data')
os.mkdir('chest_xray_fulldataset/validation_data')
os.mkdir('chest_xray_fulldataset/test_data') 

for fn in train_files:
    shutil.copy(fn, "chest_xray_fulldataset/training_data/")

for fn in validate_files:
    shutil.copy(fn, "chest_xray_fulldataset/validation_data/")
    
for fn in test_files:
    shutil.copy(fn, "chest_xray_fulldataset/test_data/")
print("Step10 completed")



print("Step11 started")
#Preparing datasets
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
print("Step11 completed")



print("Step12 started")
IMG_DIM = (150, 150)

train_files = glob.glob('chest_xray_fulldataset/training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[2].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('chest_xray_fulldataset/validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('/')[2].split('.')[0].strip() for fn in validation_files]

print('Training dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)
print("Step12 completed")



print("Step13 started")
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])
print("Step13 completed")


print("Step14 started")
batch_size = 30
num_classes = 2
epochs = 30
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[1495:1505], train_labels_enc[1495:1505])
print("Step14 completed")


print("Step15 started")

if(network==1):
    from keras.applications import vgg16
    cnn_network = vgg16.VGG16(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("VGG16 model selected.")

if(network==2):
    from keras.applications import vgg19
    cnn_network = vgg19.VGG19(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("VGG19 model selected.")

if(network==3):
    from keras.applications import resnet50
    cnn_network = resnet50.ResNet50(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("ResNet50 model selected.")

if(network==4):
    from keras.applications import inception_v3
    cnn_network = inception_v3.InceptionV3(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("InceptionV3 model selected.")

if(network==5):
    from keras.applications import xception
    cnn_network = xception.Xception(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("Xception model selected.")

if(network==6):
    from keras.applications import densenet
    cnn_network = densenet.DenseNet201(include_top=False, weights='imagenet', 
                                         input_shape=input_shape)
    print("DenseNet model selected.")

output = cnn_network.layers[-1].output
output = keras.layers.Flatten()(output)
cnn_model = Model(cnn_network.input, output)
print("Step15 completed")



print(type(index))
if(index==1):
    print("Step16 started")
    print("Processing Transfer Learning by Feature Extraction methodology")
    cnn_model.trainable = False
    for layer in cnn_model.layers:
        layer.trainable = False
    
    pd.set_option('max_colwidth', -1)
    layers = [(layer, layer.name, layer.trainable) for layer in cnn_model.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
    
    def get_bottleneck_features(model, input_imgs):
        features = model.predict(input_imgs, verbose=0)
        return features
    
    train_features_cnn = get_bottleneck_features(cnn_model, train_imgs_scaled)
    validation_features_cnn = get_bottleneck_features(cnn_model, validation_imgs_scaled)

    print('Train Bottleneck Features:', train_features_cnn.shape, 
          '\tValidation Bottleneck Features:', validation_features_cnn.shape)
    
    input_shape = cnn_model.output_shape[1]

    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    model.summary()
    
    history = model.fit(x=train_features_cnn, y=train_labels_enc,
                        validation_data=(validation_features_cnn, validation_labels_enc),
                        batch_size=10,
                        epochs=30,
                        verbose=1)
    if os.path.isfile('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_basic_cnn.h5'):        
        os.remove('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_basic_cnn.h5')
    model.save('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_basic_cnn.h5')
    print("Step16 completed")


print(type(index))
if(index==2):
    print("Step16 started")
    print("Processing Transfer Learning by Feature Extraction and Image Augmentation methodology")
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
    val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
    
    model = Sequential()
    model.add(cnn_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
              
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=20,
                                  validation_data=val_generator, validation_steps=50, 
                                  verbose=1) 
    if os.path.isfile('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_img_aug_cnn.h5'):        
        os.remove('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_img_aug_cnn.h5')
        
    model.save('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_img_aug_cnn.h5')
    
    print("Step16 completed")



print(type(index))
if(index==3):
    print("Step16 started")
    print("Processing Transfer Learning with Fine-tuning and Image Augmentation methodology")
    cnn_model.trainable = True

    set_trainable = False
    for layer in cnn_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
        
    layers = [(layer, layer.name, layer.trainable) for layer in cnn_model.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
    val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)

    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
    from keras.models import Sequential
    from keras import optimizers

    model = Sequential()
    model.add(cnn_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])
              
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=60,
                                  validation_data=val_generator, validation_steps=50, 
                                  verbose=1)
    if os.path.isfile('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_finetune_img_aug_cnn.h5'):        
        os.remove('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_finetune_img_aug_cnn.h5')    
    model.save('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_finetune_img_aug_cnn.h5')
    print("Step16 completed")


print("Step17 started")
def plot_learning_curves(history):            
    plt.plot(history.history['val_acc'])
    plt.title('Validation accuracy', size=16)
    plt.ylabel( 'Accuracy', size=14)
    plt.xlabel('No.of Epoch', size=14)    
    
    plt.tight_layout()
    
plot_learning_curves(history)
print("Step17 completed")



print("Step18 started")
# load other configurations
input_shape = (150, 150, 3)
num2class_label_transformer = lambda l: ['NORMAL' if x == 0 else 'PNEUMONIA' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'NORMAL' else 1 for x in l]
print("Step18 completed")



print("Step19 started")
IMG_DIM = (150, 150)

test_files = glob.glob('chest_xray_fulldataset/test_data/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('/')[2].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
print(test_labels[0:5], test_labels_enc[0:5])
print("Step19 completed")


if(index==1):
    print("Step20 started")
    tl_cnn = load_model('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_basic_cnn.h5')
    test_bottleneck_features = get_bottleneck_features(cnn_model, test_imgs_scaled)
    predictions = tl_cnn.predict_classes(test_bottleneck_features, verbose=0)      
    print("Step20 completed")


if(index==2):
    print("Step20 started")
    tl_img_aug_cnn = load_model('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_img_aug_cnn.h5')
    predictions = tl_img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)              
    print("Step20 completed")


if(index==3):
    print("Step20 started")
    tl_img_aug_finetune_cnn = load_model('chest_xray_fulldataset/NORMAL_PNEUMONIA_tlearn_finetune_img_aug_cnn.h5')
    predictions = tl_img_aug_finetune_cnn.predict_classes(test_imgs_scaled, verbose=0)
    print("Step20 completed")

print("Step21 started")
from PIL import Image
from random import randint
f, ax = plt.subplots(2, 4, figsize=(15,15))
for k in range(2):
    for i in range(4):
        j = randint(0,(len(test_files)-1))
        ax[k][i].imshow(Image.open(test_files[j]).resize((200, 200), Image.ANTIALIAS))        
        ax[k][i].text(10, 165, 'Actual: %s' % test_labels[j], color='k', backgroundcolor='white', alpha=0.8)
        ax[k][i].text(10, 190, 'Predicted: %s' % num2class_label_transformer(predictions[j]), color='k', backgroundcolor='white', alpha=0.8)
plt.show()
print("Step21 completed")

print("Step22 started")
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

conf_mtx = confusion_matrix(test_labels_enc, predictions) 
plot_confusion_matrix(conf_mtx, figsize=(8,6), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()
print("Step22 completed")


print("Step23 started")
def accuracy(np_array, list1):
    acc=0
    k=len(np_array)
    for i in range(k):
        if(np_array[i][0]==list1[i]):
            acc=acc+1
    accu1 = (acc/k)*100
    print("Accuracy of Prediction:",accu1,"%")
    return accu1
accu = accuracy(predictions,test_labels_enc)
print("Step23 completed")


print("Step24 started")
# Creates a list containing 6 lists, each of 3 items, all set to 0
acc_data[network-1][index-1]=accu
print("Step24 completed")

#Data captured from multiple runs and updated the acc_data array
acc_data=[[84.75, 92.15, 87.17],[76.118, 91.42, 88.52],[79.82, 90.11, 63.17],[78.217, 88.521,68.04],[83.594, 92.57,67.46],[81.56, 88.98, 56.66]]


print("Step25 started")
team_list1=["VGG16", "VGG19", "ResNet50", "InceptionV3", "Xception", "DenseNet"]
team_list2=["conv method", "Image augm", "Fine Tuning"]
pd.DataFrame(acc_data, team_list1, team_list2)


print("Step26 started")
con_method=[]
image_aug=[]
fine_tune=[]
for i in range(len(acc_data)):
    con_method.append(int(acc_data[i][0]))
    image_aug.append(int(acc_data[i][1]))
    fine_tune.append(int(acc_data[i][2]))
print("Step26 completed")


print("Step27 started")
labels = ['VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'Xception', 'DenseNet']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10,5))
rects1 = ax.bar(x - width/2, con_method, width, label='Feature Extraction')
rects2 = ax.bar(x + width/2, image_aug, width, label='Feature Extraction with Image augmentation')
rects3 = ax.bar(x + 1.5*width, fine_tune, width, label='Fine-tuning and Image augmentation')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy', size=14)
ax.set_title('Accuracy by Pre-trained Convolutional neural network and methodology', size=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='center left',bbox_to_anchor=(1,0.5))
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
print("All steps completed")





