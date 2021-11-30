from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics  import AUC, CategoricalAccuracy, FalsePositives
from numpy import array, max, argmax
import properties
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model=Sequential()
model.add(Conv2D(filters=properties.filters[0],
kernel_size=properties.kernel_size[0],activation='relu',
padding='same',input_shape=properties.input_shape,strides=properties.stride[0],
dilation_rate=properties.dilation[0]))
model.add(MaxPool2D(pool_size=properties.pool_size[0]))
for i in range(1,len(properties.filters)):
    model.add(Conv2D(filters=properties.filters[i],kernel_size=properties.kernel_size[i],activation='relu',padding='same',strides=properties.stride[i],dilation_rate=properties.dilation[i]))
    model.add(MaxPool2D(pool_size=properties.pool_size[i]))
model.add(Flatten())
for i in range(len(properties.dense_layer_neurons)):
    if i!=len(properties.dense_layer_neurons):
        model.add(Dense(units=properties.dense_layer_neurons[i],activation='relu'))
        model.add(Dropout(properties.dropout_probability))
    else:
        model.add(Dense(units=properties.dense_layer_neurons[i],activation='softmax'))
model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=CategoricalCrossentropy(),
metrics=[AUC(),CategoricalAccuracy(),FalsePositives()])

print(model.summary())
