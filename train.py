import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

import tensorflow
# to divide our data into train and validation set
from sklearn.model_selection import train_test_split
#to encode our labels
from tensorflow.keras.utils import to_categorical
#to build our model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
# Our optimizer options
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
#Callback options
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
#importing image data generator for data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(epoch=30, model_name="my_model.hdf5"):
    train =pd.read_csv('dataset/mnist_train.csv')
    y_train=train['label']
    x_train=train.drop('label',axis=1).values
    x_train=x_train.reshape(60000,28,28)
    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.10,random_state=19)

    print(f'The training set has {x_train.shape[0]} images and the validation set has {x_val.shape[0]} images')

    y_cat_train=to_categorical(y_train,num_classes=10)
    y_cat_val=to_categorical(y_val,num_classes=10)

    # random_image=x_train[78]
    # random_label=y_train[78]
    # plt.imshow(random_image,cmap='binary')
    # plt.title(random_label,fontsize=20,weight='bold',color='red')
    # plt.show()


    x_train_scaled = x_train / 255
    x_val_scaled = x_val / 255

    # plt.imshow(x_train_scaled[78],cmap='binary')
    # plt.title(y_train[78],fontsize=20,weight='bold',color='red')
    # plt.show()

    x_train_final = x_train_scaled.reshape(x_train_scaled.shape[0],28,28,1)
    x_val_final = x_val_scaled.reshape(x_val_scaled.shape[0],28,28,1)

    # RMSProp or Adam
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),input_shape=(28,28,1),padding='Same',activation='relu'))
    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),input_shape=(28,28,1),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1),padding='Same',activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    image_gen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,fill_mode='nearest')

    train_image_gen = image_gen.fit(x_train_final)

    epoch = 30

    model.fit_generator(image_gen.flow(x_train_final,y_cat_train),epochs=epoch,validation_data=(x_val_final,y_cat_val),callbacks=[learning_rate_reduction])
    model.save(model_name)

    # Plot the accuracy and loss respect to epoch
    metrics=pd.DataFrame(model.history.history)
    metrics.to_csv(f"pretrained/{model_name}.csv")
    metrics[['loss' , 'val_loss']].plot()
    plt.show()
    metrics[['accuracy' , 'val_accuracy']].plot()
    plt.show()

    data = []
    test=pd.read_csv('dataset/mnist_test.csv', usecols=range(1, 785))
    label = pd.read_csv("dataset/mnist_test.csv", usecols=[0])
    x_test = test.values
    x_test = x_test.reshape(10000, 28, 28)
    x_test = x_test / 255
    predict = model.predict(x_test.reshape(10000, 28, 28, 1))
    for i, p in enumerate(predict):
        data.append([np.argmax(p), label.at[i, "label"],
                    True if np.argmax(p) == int(label.at[i, "label"])
                    else False])
    df = pd.DataFrame(data, columns=["Predicted Value", "True Value", "Accurate"])
    df.to_csv(f"pretrained/{model_name} predict.csv")
