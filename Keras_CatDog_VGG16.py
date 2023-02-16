# encoding=utf-8
# Programmer: 劉志俊
# Date: 2021/11/17
# 使用 VGG16 辨識 Cat 與 Dog
# 版本: v1
# 訓練準確率 0.9081
# 測試準確率 0.9062
# 執行時間: 每個 Epoch 50 秒, 10 Epochs 需時 11 分

from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# 產生影像維度
img_width, img_height = 224, 224
num_classes = 2                              # 最終輸出類型個數: 2

train_data_dir = 'ImageGen/train'            # 訓練影像所在子目錄
validation_data_dir = 'ImageGen/validation'  # 驗證影像所在子目錄
image_aug  = 1                               # 影像擴增倍率
nb_train_samples = 200 * image_aug           # 訓練影像樣本數
nb_validation_samples = 100 * image_aug      # 驗證影像樣本數

# 可調整網路模型超參數()
epochs = 10                                  # 訓練世代
batch_size = 64                              # 批次訓練影像張數
trainable = True                             # 遷移學習, 是否凍結預訓練模型權重
learning_rate = 0.01                         # 學習率
dropout = 0.5                                # 丟棄率

## 遷移學習 Transfer Learning
# 建立模型
# 使用預先建立之 VGG16 模型
base_model = VGG16(weights='imagenet', include_top=False, pooling='none')

# 印出模型中各層網路名稱
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 加入 GlobalAveragePooling2D 層
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 加入全連接層
x = Dense(1024, activation='relu')(x)
# 加入最終分類層
predictions = Dense(num_classes, activation='softmax')(x)

# 設定模型
model = Model(inputs=base_model.input, outputs=predictions)

# 僅訓練 top layers 
# 凍結預先建立之 InceptionResNetV2 的 Convolution layers
for layer in base_model.layers:
    layer.trainable = trainable

print(model.summary())

# 選擇最佳化器 Optimizer
sgd = SGD(lr=learning_rate)   # learning rate: 先試 0.01, 
                              # 看訓練結果再試 0.001 或 0.1, 
				   	          # 看訓練結果再試 0.0001 或 1
# rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)  
# adagrad = Adagrad(lr=0.005, epsilon=None, decay=0.0)       
# adadelta = Adadelta(lr=0.8, rho=0.95, epsilon=None, decay=0.0)
# adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)                          
# adamax = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# 編譯模型                          
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

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

