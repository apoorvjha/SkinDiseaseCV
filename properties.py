import os

rotation_range=90
steps=15
dimension=(32,32)
kernel_size=3
input_shape=(dimension[0],dimension[1],3)
stride=(1,1)
dilation=(1,1)
pool_size=(2,2)
dropout_probability=0.2
learning_rate=1e-4
batch_size=64
epochs=12
validation_split=0.1
test_ratio=0.1
random_state=42
verbose=1
dataset_path='./static/Dataset_Dermat/'
model_name="model_VGG19_dermat.h5"
disease=os.listdir(dataset_path)
classes={disease[i] : i for i in range(len(disease)) }
precision=4
shuffle=True
decay_steps=1000000
decay_rate=0.1
n_output=len(disease)