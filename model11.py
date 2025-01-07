import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random 
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import pickle
model_path="model_save.h5"
label_binarizer_path="label_binarizer.pkl"

dir = r"C:\Users\VICTUS\Downloads\archive\Train"
train_data = []
img_size = 32
non_chars = ["#","$","&","@"]
for i in os.listdir(dir):
     if i in non_chars:
         continue
     count = 0
     sub_directory = os.path.join(dir,i)
     for j in os.listdir(sub_directory):
         count+=1
         if count > 4000:
             break
         img = cv2.imread(os.path.join(sub_directory,j),0)
         img = cv2.resize(img,(img_size,img_size))
         train_data.append([img,i])
        
print(len(train_data))

val_dir = r"C:\Users\VICTUS\Downloads\archive\Validation"
val_data = []
img_size = 32
for i in os.listdir(val_dir):
     if i in non_chars:
         continue
     count = 0
     sub_directory = os.path.join(val_dir,i)
     for j in os.listdir(sub_directory):
         count+=1
         if count > 1000:
             break
         img = cv2.imread(os.path.join(sub_directory,j),0)
         img = cv2.resize(img,(img_size,img_size))
         val_data.append([img,i])
        
print(len(val_data)) 


random.shuffle(train_data)
random.shuffle(val_data)

train_X = []
train_Y = []
for features,label in train_data:
     train_X.append(features)
     train_Y.append(label)
    
val_X = []
val_Y = []
for features,label in val_data:
     val_X.append(features)
     val_Y.append(label)


LB = LabelBinarizer()
train_Y = LB.fit_transform(train_Y)
val_Y = LB.fit_transform(val_Y)

train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,32,32,1)
train_Y = np.array(train_Y)


val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1,32,32,1)
val_Y = np.array(val_Y)

print(train_X.shape,val_X.shape)

print(train_Y.shape,val_Y.shape)




model = Sequential()
 # Define the input layer explicitly
model.add(Input(shape=(32, 32, 1)))

 # Add the rest of the layers
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(35, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy']) 
history = model.fit(train_X,train_Y, epochs=35, batch_size=32, validation_data = (val_X, val_Y),  verbose=1)
with open("label_binarizer.pkl","wb") as file:
    pickle.dump(LB,file)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


## for save model
model.save(model_path)
with open(label_binarizer_path,"wb") as file:
    pickle.dump(LB,file)

## for load model
# model=load_model(model_path)
# with open(label_binarizer_path,"rb") as file:
#     LB=pickle.load(file)
# def sort_contours(cnts, method="left-to-right"):
#     reverse = False
#     i = 0
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#     key=lambda b:b[1][i], reverse=reverse))
#     # return the list of sorted contours and bounding boxes
#     return (cnts, boundingBoxes)



# def get_letters(img):
#     letters = []
#     image = cv2.imread(img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
#     dilated = cv2.dilate(thresh1, None, iterations=2)
    
#     cv2.imshow("Thresholded Image", thresh1)
#     cv2.imshow("Dilated Image", dilated)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#     cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sort_contours(cnts, method="left-to-right")[0]
#     # loop over the contours
#     for c in cnts:
#         if cv2.contourArea(c) > 10:
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi = gray[y:y + h, x:x + w]
#         thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#         thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
#         thresh = thresh.astype("float32") / 255.0
#         thresh = np.expand_dims(thresh, axis=-1)
#         thresh = thresh.reshape(1,32,32,1)
#         ypred = model.predict(thresh)
#         ypred = LB.inverse_transform(ypred)
#         [x] = ypred
#         letters.append(x)
#     return letters, image

# def get_word(letter):
#     word = "".join(letter)
#     return word

# letter,image = get_letters(r"C:\Users\VICTUS\Pictures\Screenshots\jincy.png")
# word = get_word(letter)
# print(word)
# plt.imshow(image)