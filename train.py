import Model
import DataAugmentation
import properties

path='./static/Dataset/'
X=[]
Y=[]
directories=list(properties.classes.keys())
for i in range(len(directories)):
    image=DataAugmentation.Augment(path+directories[i]+'/')
    for j in range(image.shape[0]):
        Y.append(directories[i])
    X.extend(image)
X=Model.array(X)
Y=Model.Class2OHV(Model.array(Y))
model=Model.instantiateModel()
model,history=Model.fit(X,Y,model)
Model.save(model)    
Model.plot_train_history(history)
