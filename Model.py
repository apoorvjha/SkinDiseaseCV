from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics  import AUC, CategoricalAccuracy, FalsePositives, Accuracy
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard
from numpy import array, max, argmax, zeros
import properties
from matplotlib import pyplot as plt



def instantiateModel(mode=1):
    input_shape=properties.input_shape
    n_output=properties.n_output
    dropout_p=properties.dropout_probability
    learning_rate=properties.learning_rate
    #lr_schedule = ExponentialDecay(
    #initial_learning_rate=learning_rate,
    #decay_steps=properties.decay_steps,
    #decay_rate=properties.decay_rate)
    lr_schedule = learning_rate
    if mode == 0:
        kernel_n=properties.kernel_size
        stride=properties.stride      # skips, kernel makes at every convolution
        dilation=properties.dilation  # kernel coverage
        pooling_size=properties.pool_size
        model=Sequential()
        model.add(Conv2D(filters=16,kernel_size=kernel_n,activation='relu', padding='same',input_shape=input_shape,strides=stride, dilation_rate=dilation))
        model.add(MaxPool2D(pool_size=pooling_size))
        model.add(Conv2D(filters=32,kernel_size=kernel_n,activation='relu',padding='same',strides=stride, dilation_rate=dilation))
        model.add(MaxPool2D(pool_size=pooling_size))
        model.add(Conv2D(filters=64,kernel_size=kernel_n,activation='relu',padding='same',strides=stride, dilation_rate=dilation))
        model.add(MaxPool2D(pool_size=pooling_size))
        model.add(Conv2D(filters=128,kernel_size=kernel_n,activation='relu',padding='same',strides=stride, dilation_rate=dilation))
        model.add(MaxPool2D(pool_size=pooling_size))
        model.add(Flatten())
        model.add(Dense(units=input_shape[0] * 128,activation='relu'))
        model.add(Dropout(dropout_p))
        model.add(Dense(units=128,activation='relu'))
        model.add(Dropout(dropout_p))
        model.add(Dense(units=64,activation='relu'))
        model.add(Dropout(dropout_p))
        model.add(Dense(units=32,activation='relu'))
        model.add(Dropout(dropout_p))
        model.add(Dense(units=16,activation='relu'))
        model.add(Dropout(dropout_p))
        model.add(Dense(units=n_output,activation="softmax"))
        model.compile(optimizer=Adam(learning_rate=lr_schedule),loss=CategoricalCrossentropy(),metrics=[AUC(),CategoricalAccuracy(),FalsePositives()])
        return model
    elif mode == 1:
        vgg_top = VGG19(weights='imagenet',input_shape=input_shape,classes=n_output,include_top=False)
        for layer in vgg_top.layers:
            layer.trainable=False
        vgg_fc = Flatten() (vgg_top.output)
        #vgg_fc = Dense(units=512,activation='relu')(vgg_fc)
        #vgg_fc = Dropout(dropout_p)(vgg_fc)
        #vgg_fc = Dense(units=256,activation='relu')(vgg_fc)
        #vgg_fc = Dropout(dropout_p)(vgg_fc)
        vgg_fc = Dense(units=128,activation='relu')(vgg_fc)
        vgg_fc = Dropout(dropout_p)(vgg_fc)
        vgg_fc = Dense(units=64,activation='relu')(vgg_fc)
        vgg_fc = Dropout(dropout_p)(vgg_fc)
        vgg_fc = Dense(units=32,activation='relu')(vgg_fc)
        vgg_fc = Dropout(dropout_p)(vgg_fc)
        vgg_out = Dense(units=n_output,activation='softmax')(vgg_fc)  
        model = Model(inputs=vgg_top.input,outputs=vgg_out)
        model.compile(optimizer=Adam(learning_rate=lr_schedule),loss=CategoricalCrossentropy(),metrics=[AUC(),CategoricalAccuracy(), Accuracy()])
        return model
    else:
        resnet_top = ResNet50(weights='imagenet',input_shape=input_shape,classes=n_output,include_top=False)
        for layer in resnet_top.layers:
            layer.trainable=False
        resnet_fc = Flatten() (resnet_top.output)
        resnet_fc = Dense(units=2048,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_fc = Dense(units=512,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_fc = Dense(units=256,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_fc = Dense(units=128,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_fc = Dense(units=64,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_fc = Dense(units=32,activation='relu')(resnet_fc)
        resnet_fc = Dropout(dropout_p)(resnet_fc)
        resnet_out = Dense(units=n_output,activation='softmax')(resnet_fc)  
        model = Model(inputs=resnet_top.input,outputs=resnet_out)
        #print(model.summary())
        model.compile(optimizer=Adam(learning_rate=lr_schedule),loss=CategoricalCrossentropy(),metrics=[AUC(),CategoricalAccuracy(),FalsePositives()])
        return model
def fit(X,Y,model):
    # To launch the tensorboard ->  tensorboard --logdir=./TB_logs
    #tensorboard_cb=TensorBoard(log_dir="./TB_logs")
    start=time.time()
    history=model.fit(X,Y,batch_size=properties.batch_size,epochs=properties.epochs,validation_split=properties.validation_split)#,verbose=properties.verbose)#,callbacks=[tensorboard_cb])
    print(time.time()-start)
    return model,history
def save(model):
    model.save(properties.model_name)
def load():
    return load_model(properties.model_name)
def OHV2Class(predictions):
    class_vector=[]
    for i in range(predictions.shape[0]):
        class_vector.append(argmax(predictions[i]))
    return array(class_vector)
def Class2OHV(Y):
    OHV_vector=[]
    for i in Y:
        temp=zeros((properties.n_output),dtype='uint8')
        temp[properties.classes[i]]=1
        OHV_vector.append(temp)
    return array(OHV_vector)
def plot_train_history(history):
    X=history.history['loss']
    Y=history.history['val_loss']
    title="Loss" 
    label=["Metrics","Epochs"]
    legend=["Train","Validation"]
    plt.plot(X)
    plt.plot(Y)
    plt.title(title)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(legend,loc='upper left')
    plt.savefig("Loss.png")
    plt.close()
    X=history.history['accuracy']
    Y=history.history['val_accuracy']
    title="Accuracy"
    label=["Metrics","Epochs"]
    plt.plot(X)
    plt.plot(Y)
    plt.title(title)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(legend,loc='upper left')
    plt.savefig("accuracy.png")
    plt.close()
def predict(X,model):
    return model.predict(X)

