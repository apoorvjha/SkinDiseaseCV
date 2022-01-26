from flask import Flask, render_template, redirect, flash, request, url_for, session
from werkzeug.utils import secure_filename
import json
import Auth
import time
import Model
import ImageProcessing as driver
import DataAugmentation
import properties
import os
from os import listdir
from sklearn.model_selection import train_test_split

app=Flask(__name__)
app.secret_key="qazwsx@2022"

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route('/logout')
def logout():
    if session['username']:
        session.pop('username',None)
        session.pop('email',None)
        session.pop('profilePic',None)
        session.pop('type',None)
        session.pop('userId',None)        
        flash("Successfully logged out!",'alert alert-success')        
    return redirect(url_for('index'))
@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/loginBack',methods=['POST'])
def login_back():
    if request.method=='POST':
        userId=request.form['userId']
        password=request.form['password']
        status,res,msg,category=Auth.checkCredentials(userId,password)
        flash(msg,category)
        if status==True:
            session['username']=res[0][0]
            session['email']=res[0][1]
            session['profilePic']=res[0][2]
            session['type']=res[0][3]
            session['userId']=res[0][4]
            return redirect(url_for('index'))
        else:
            return redirect(url_for('login'))
    else:
        flash("Unsupported method of request! Please use POST instead.",'alert alert-danger')
        return redirect(url_for('index'))
@app.route('/registerBack',methods=['POST'])
def registerBack():
    if request.method=='POST':
        userId=request.form['userId']
        password=request.form['password']
        email=request.form['email']
        profilePic=request.files['profilePic']
        fileName=userId + secure_filename(profilePic.filename)
        profilePic.save('static/PROFILE_PIC/'+fileName)
        profilePath='static/PROFILE_PIC/'+fileName
        status=Auth.register(userId,password,email,profilePath)
        if status==True:
            # as we advance, incorporate functionality of OTP verification as well.
            flash("You are successfully registered!.",'alert alert-success') 
            return redirect(url_for('index'))
        else:
            flash("User with provided data already exists! ",'alert alert-danger')
            return redirect(url_for('register'))
    else:
        flash("Unsupported method of registration! Please use the registration tab instead.",'alert alert-danger')
        return redirect(url_for("HomePage"))
@app.route('/users')
def getUsers():
    if session['type']=='admin':
        response={"users" : []}
        users=Auth.getUsers()
        for i in users:
            response["users"].append({"userID" : i[0],"profilePic" : i[1], "username" : i[2], "email" : i[3],"isActive" : i[4]})
        return response
    else:
        #flash("You donot have access to this endpoint!","alert alert-danger")
        return {"status" : 404}

@app.route('/activate/<id>')
def activate(id):
    if session['type']=='admin':
        Auth.activate(id)
        #flash("User activated successfully!","alert alert-success")
        return {"status" : 200}
    else:
        #flash("You donot have access to this endpoint!","alert alert-danger")
        return {"status" : 404}

@app.route('/deactivate/<id>')
def deactivate(id):
    if session['type']=='admin':
        Auth.deactivate(id)
        #flash("User deactivated successfully!","alert alert-success")
        return {"status" : 200}
    else:
        #flash("You donot have access to this endpoint!","alert alert-danger")
        return {"status" : 404}
@app.route('/settings')
def settings():
    return render_template('settings.html')
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/changeCredentials',methods=['POST'])
def changeCredentials():
    if session['userId']:
        if request.method=='POST':
            mode=request.form['mode']
            value=request.form["value"]
            if mode==0:
                Auth.changeUserName(session['userId'],value)
            elif mode==1:
                Auth.changePassword(session['userId'],value)
            else:
                Auth.changeEmail(session['userId'],value)
            return {"status" : 200}
        else:
            return {"status" : 404}
    else:
        return {"status" : 500}

@app.route('/changeProfilePicture',methods=['POST'])
def changeProfPic():
    if session['userId']:
        if request.method=='POST':
            # change profile pic.
            profilePic=request.files['profpic']	
            fileName=session['profilePic']
            profilePic.save(fileName)
            flash("Profile pic changed successfully!","alert alert-success")
            return redirect(url_for('index'))
        else:
            return {"status" : 404}
    else:
        return {"status" : 500}

@app.route('/inference',methods=['POST'])
def inference():
    if session['userId']:
        if request.method=='POST':
            image=request.files['file']	
            fileName=str(session['userId']) + secure_filename(image.filename)
            image.save('static/processing_dir/'+fileName)
            start=time.time()
            X=[]
            image_cv=driver.read_image('static/processing_dir/'+fileName)
            image_cv=driver.Resize(image_cv,dimension=properties.dimension)
            image_cv=driver.SkinDetection(image_cv)
            X.append(image_cv)
            X=Model.array(X)
            model=Model.load()
            predictions=Model.OHV2Class(Model.predict(X,model))
            time_taken=time.time() - start
            return {"image" : 'static/processing_dir/'+fileName,"prediction" : list(properties.classes.keys())[predictions[0]], "status" : 200,"response_time" : round(time_taken,3)}
        else:
            return {"status" : 404}
    else:
        return {"status" : 500}
@app.route('/getModelStats')
def getModelStats():
    if session['userId']:
        if request.method=='GET':
            response={"params" : [],"status" : 200}
            response["params"].append({"name" : "rotation_range","value" : properties.rotation_range})
            response["params"].append({"name" : "steps","value" : properties.steps})
            #response["params"].append({"name" : "dimension","value" : properties.dimension})
            #response["params"].append({"name" : "kernel_size","value" : properties.kernel_size})
            #response["params"].append({"name" : "input_shape","value" : properties.input_shape})
            #response["params"].append({"name" : "stride","value" : properties.stride})
            #response["params"].append({"name" : "dilation","value" : properties.dilation})
            #response["params"].append({"name" : "pool_size","value" : properties.pool_size})
            response["params"].append({"name" : "n_output","value" : properties.n_output})
            response["params"].append({"name" : "dropout_probability","value" : properties.dropout_probability})
            response["params"].append({"name" : "learning_rate","value" : properties.learning_rate})
            response["params"].append({"name" : "batch_size","value" : properties.batch_size})
            response["params"].append({"name" : "epochs","value" : properties.epochs})
            response["params"].append({"name" : "model_name","value" : properties.model_name})
            response["params"].append({"name" : "validation_split","value" : properties.validation_split})
            response["params"].append({"name" : "test_ratio","value" : properties.test_ratio})
            response["params"].append({"name" : "random_state","value" : properties.random_state})
            response["params"].append({"name" : "verbose","value" : properties.verbose})
            #response["params"].append({"name" : "classes","value" : properties.classes})
            response["params"].append({"name" : "precision","value" : properties.precision})
            response["params"].append({"name" : "shuffle","value" : properties.shuffle})
            response["params"].append({"name" : "decay_steps","value" : properties.decay_steps})
            response["params"].append({"name" : "decay_rate","value" : properties.decay_rate})
            return response
        else:
            return {"status" : 404}
    else:
        return {"status" : 500}

@app.route('/getDataset')
def getDataset():
    if session['userId']:
        if request.method=='GET':
            response={"datasets" : [],"status" : 200}
            path='./static/Dataset/'
            directories=list(properties.classes.keys())
            for i in range(len(directories)):
                for j in listdir(path+directories[i]+'/'):
                    response["datasets"].append({"name" : directories[i],"image" : path+directories[i]+'/'+j})
            return response
        else:
            return {"status" : 404}
    else:
        return {"status" : 500}

if __name__=='__main__':
    app.run(debug=True,port=5000)