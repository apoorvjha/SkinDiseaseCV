import Model
import DataAugmentation
import properties
from sklearn.model_selection import train_test_split
import os

path='./static/Dataset/'
X=[]
Y=[]
directories=list(properties.classes.keys())
for i in range(len(directories)):
    image=DataAugmentation.Augment(path+directories[i]+'/')
    for j in range(image.shape[0]):
        Y.append(directories[i])
        X.append(image[j])
X=Model.array(X)
Y=Model.Class2OHV(Model.array(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=properties.test_ratio,shuffle=properties.shuffle,random_state=properties.random_state)
if os.path.exists(properties.model_name):
    model=Model.load()
else: 
    model=Model.instantiateModel()
model,history=Model.fit(X_train,Y_train,model)
Model.save(model)    
Model.plot_train_history(history)
predictions=Model.predict(X_test, model)
print("Accuracy = "+str(Model.accuracy(predictions, Y_test)))
