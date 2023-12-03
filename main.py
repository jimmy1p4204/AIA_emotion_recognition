import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
 

#
# 建立訓練物件與測試物件
#
def getValidationGenerator():
    val_dir = 'test'
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')
    return validation_generator

def getTrainGenerator():
    train_dir = 'train'
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')
    return train_generator


#
# 設定情緒訓練 Model 參數
#
def getEmotionModelSetting():
    emotion_model = Sequential()
    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))
    return emotion_model

#
# 訓練模組，產生情緒辨識 model
#
import tensorflow as tf
import PIL.Image
import scipy

def trainModel():
    # optimizer = tf.keras.optimizers.legacy.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    emotion_model = getEmotionModelSetting()
    emotion_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    emotion_model_info = emotion_model.fit_generator(
            getTrainGenerator(),
            steps_per_epoch=28709 // 64,
            epochs=50,
            validation_data=getValidationGenerator(),
            validation_steps=7178 // 64)
    return emotion_model

# Saving the model
def save(emotion_model, saveFileName):
    emotion_model.save(saveFileName)

# 載入剛剛訓練好的情緒辨識 model
from keras.models import load_model
def load(modelName):
    emotion_model = load_model(modelName)
    return emotion_model

# 呈現情緒辨識結果 - 直方圖
def show_analysis_result(emotions):
  objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
  y_pos = np.arange(len(objects))
  
  plt.bar(y_pos, emotions, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('percentage')
  plt.title('emotion')
  
  plt.show()

#
# 進行情緒辨識
#
import cv2
            
def facecrop(image):  
    facedata = 'haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(facedata)

    print(image)

    img = cv2.imread(image)
    
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        sub_face = img[y:y+h, x:x+w]

        
        cv2.imwrite('capture.jpg', sub_face)
        #print ("Writing: " + image)


#
# 辨識情緒
#
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
def regnize(emotion_model, fileName):
    true_image = image.load_img(fileName)
    img = image.load_img(fileName, color_mode="grayscale", target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = emotion_model.predict(x)
    show_analysis_result(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    plt.imshow(true_image)
    plt.show()


if __name__ == '__main__':
    # model = trainModel()
    # save(mode, 'model.h5')

    model = load('model.h5')

    facecrop('photo.png')
    regnize(model, 'capture.jpg')




