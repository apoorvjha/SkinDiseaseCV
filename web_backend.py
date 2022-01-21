from flask import Flask, render_template, redirect, flash, request, url_for, session
from werkzeug.utils import secure_filename
import Auth

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


if __name__=='__main__':
    app.run(debug=True,port=5000)