import time
start=time.time()

import Model
import ImageProcessing as driver
import properties
import os

path='./static/processing_dir/'
X=[]
for i in os.listdir(path):
    image=driver.read_image(path+i)
    image=driver.Resize(image,dimension=properties.dimension)
    image=driver.SkinDetection(image)
    X.append(image)
X=Model.array(X)
if os.path.exists(properties.model_name):
    model=Model.load()
else: 
    import train
    model=Model.load()
predictions=Model.OHV2Class(Model.predict(X,model))
for i in range(len(os.listdir(path))):
    print(f"{os.listdir(path)[i]} belongs to {list(properties.classes.keys())[predictions[i]]}")

print(time.time()-start)   


