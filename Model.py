from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics  import AUC, CategoricalAccuracy, FalsePositives
from numpy import array, max, argmax
import properties
from os import environ
from matplotlib import pyplot as plt

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def instantiateModel():
    kernel_n=properties.kernel_size
    input_shape=properties.input_shape
    stride=properties.stride      # skips, kernel makes at every convolution
    dilation=properties.dilation  # kernel coverage
    pooling_size=properties.pool_size
    dropout_p=properties.dropout_probability
    n_output=properties.n_output
    learning_rate=properties.learning_rate
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
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss=CategoricalCrossentropy(),metrics=[AUC(),CategoricalAccuracy(),FalsePositives()])
    return model
def fit(X,Y,model):
    history=model.fit(X,Y,batch_size=properties.batch_size,epochs=properties.epochs,validation_split=properties.validation_split,verbose=properties.verbose)
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
        OHV_vector.append(properties.classes[i])
    return array(OHV_vector)
def accuracy(predicted,actual,mode=0):
    assert predicted.shape==actual.shape and mode in [0,1] , "Supplied parameters are not valid!"
    # mode=0 ; supplied parameters are OHVs
    # mode=1 ; supplied parameters are class integer mappings.
    count=0
    if mode==0:
        for i in range(predicted.shape[0]):
            if argmax(predicted[i])==argmax(actual[i]):
                count+=1
        return round(count * 100 / predicted.shape[0],properties.precision)
    else:
        for i in range(predicted.shape[0]):
            if predicted[i]==actual[i]:
                count+=1
        return round(count * 100 / predicted.shape[0],properties.precision)
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
    X=history.history['categorical_accuracy']
    Y=history.history['val_categorical_accuracy']
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
