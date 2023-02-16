# encoding=utf-8
# Programmer: 劉志俊
# Date: 2021/11/17
# 使用 LeNet 辨識 Cat 與 Dog
# 版本: v1
# 訓練準確率 0.7169
# 測試準確率 0.6094
# 執行時間: 每個 Epoch 12 秒, 10 Epochs 需時 2 分

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 產生影像維度
num_classes = 2

train_data_dir = 'ImageGen/train'            # 訓練影像所在子目錄
validation_data_dir = 'ImageGen/validation'  # 驗證影像所在子目錄
image_aug = 1                                # 影像擴增倍率
nb_train_samples = 200 * image_aug           # 訓練影像樣本數
nb_validation_samples = 100 * image_aug      # 驗證影像樣本數

# 可調整網路模型超參數()                 # 1.1 網路模型架構(使用不同訓練程式)
trainable = True                       # 1.2 遷移學習, 是否凍結預訓練模型權重

                                       # 2.1 最佳化器(本程式設定)
learning_rate = 0.01                   # 2.2 學習率
batch_size = 64                        # 2.3 批次訓練影像張數
                                             
                                       # 3.1 頻譜類型(頻譜產生程式內設定)
                                       # 3.2 頻譜調色盤(頻譜產生程式內設定)
img_width, img_height = 112, 112       # 3.3 影像解析度

batch_normalize = True                 # 4.1 批次正規化
dropout = 0.5                          # 4.2 丟棄率
epochs = 10                            # 4.3 訓練世代

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# 建立模型
model = Sequential()
# C1層： 6個特徵卷積核, 卷積核大小 5x5, 28x28 => 6@24x24
model.add(Conv2D(6, kernel_size=(5, 5), 
                 activation='relu',
                 input_shape=input_shape))
# S2層：下採樣層, 池化大小 2x2, 6@24x24 => 6@12x12
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# C3層： 16個特徵卷積核, 卷積核大小 5x5, 6@12x12 => 16@8x8
if batch_normalize:
    model.add(BatchNormalization())
model.add(Conv2D(16, (5, 5), activation='relu'))
# S4層：下採樣層, 池化大小 2x2, 16@8x8 => 16@4x4
model.add(MaxPooling2D(pool_size=(2, 2)))
if batch_normalize:
    model.add(BatchNormalization())
model.add(Dropout(dropout))
# C5層： 120 個特徵卷積核, 卷積核大小 4x4, 16@4x4 => 120@1x1
model.add(Conv2D(120, (5, 5), activation='relu'))

model.add(Flatten())
# F6層： 全連接層, 120維轉為84維
model.add(Dense(84))
model.add(Activation("relu"))
# 輸出層： 全連接層, 84維轉為2維 分類
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 編譯模型
sgd = SGD(lr=learning_rate)   # learning rate: 先試 0.01, 
                              # 看訓練結果再試 0.001 或 0.1, 
				   	          # 看訓練結果再試 0.0001 或 1
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())

# 產生訓練資料集之擴增影像
train_datagen = ImageDataGenerator(
    rescale=1. / 255)

# 產生驗證資料集之擴增影像
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    save_to_dir='ImageGen/augmented_train',  # 擴增影像放置子目錄
    save_to_dir= None,
    save_prefix='aug',                       # 擴增影像檔名開頭
    save_format='png',                       # 影像檔格式
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    save_to_dir='ImageGen/augmented_validation',  # 擴增影像放置子目錄
    save_to_dir= None,
    save_prefix='aug',                            # 擴增影像檔名開頭
    save_format='png',                            # 影像檔格式
    shuffle=False,
    class_mode='categorical')

# 由擴增樣本訓練模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# 繪出訓練過程準確度變化
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




