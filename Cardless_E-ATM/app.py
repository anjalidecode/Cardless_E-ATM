from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
from dotenv import load_dotenv
import os
import re
import sqlite3
import pandas as pd
import numpy as np
import requests
import MySQLdb
import cv2
import face_recognition
from PIL import Image, ImageTk
import pickle
import time
import pingenerate as pingen

import random as r
import pyotp
import time 
import smtplib 
from email.message import EmailMessage
from datetime import datetime
import train_faces as knntrain

print(cv2.__version__)

load_dotenv()  # Load .env file

mydb = MySQLdb.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    passwd=os.getenv('DB_PASSWORD'),
    db=os.getenv('DB_NAME')
)

conn = mydb.cursor()

recognizer = cv2.face.LBPHFaceRecognizer_create()

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
fontColor=(255,255,255)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
socketio = SocketIO(app)
clientid=0
data=[]
data1=[]

otp = "0000"

totp = pyotp.TOTP(os.getenv('OTP_SECRET'))


def otpgen():
        otp=""
        for i in range(4):
                otp+=str(r.randint(1,9))
        print ("Your One Time Password is ")
        print (otp)
        return otp
        
@app.route('/')
#loading login page or main chat page
def index():
        print("session.get('logged_in')==",session.get('logged_in'))
        if not session.get('logged_in'):
                return render_template("login.html")
        else:
                cname=session.get('cname')
                return render_template('dashboard.html',uname=cname)


@app.route('/registerpage',methods=['POST'])
def reg_page():
    return render_template("register.html")
@app.route('/forgetpasspage',methods=['GET','POST'])
def fpass_page():
    return render_template("forgetpass.html")
        
@app.route('/loginpage',methods=['POST'])
def log_page():
    return render_template("login.html")
    
def StartCamera(cid, name):
    print("cid:", cid)
    sampleNum = 0
    img_counter = 0

    DIR = f"./Dataset/{name}_{cid}"
    try:
        os.mkdir(DIR)
        print("Directory", name, "Created")
    except FileExistsError:
        print("Directory", name, "already exists")
        img_counter = len(os.listdir(DIR))  # fixed

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Video", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"{DIR}/opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def getProfile(Id):
        data=[]
        cmd="SELECT * FROM login WHERE cid="+str(Id)
        #print(cmd)
        conn.execute(cmd)

        cursor=conn.fetchall()
        #print(cursor)
        profile=None
        for row in cursor:
                #print(row)
                
                profile=row[1]

        print("data value",data)
        
        
        conn.close
        return profile
        
        
def find_faces(image_path):

        recognizer.read('trainer/trainer.yml')
        image = cv2.imread(image_path)
        # Make a copy to prevent us from modifying the original
        color_img = image.copy()
        filename = os.path.basename(image_path)
        # OpenCV works best with gray images
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # Use OpenCV's built-in Haar classifier
        faces=faceCascade.detectMultiScale(gray_img, 1.2,5)
        print('Number of faces found: {faces}'.format(faces=len(faces)))
        if len(faces)==0:
                return 0
        else:
                for (x, y, w, h) in faces:
                        #cv2.rectangle(color_img, (x, y), (x+width, y+height), (0, 255, 0), 2)
                        cv2.rectangle(color_img,(x,y),(x+w,y+h),(225,0,0),2)
                        Id, conf = recognizer.predict(gray_img[y:y+h,x:x+w])
                        print(Id)
                        print(conf)
                        if(20<conf and conf<80):
                                #data1=data
                                profile=getProfile(Id)
                                print("cliend id",Id)
                                print("Name",profile)                   
                                cv2.putText(color_img,str(Id), (x,y+h),font, fontScale, fontColor)
                                cv2.putText(color_img,str(profile), (x,y+h+30),font, fontScale, fontColor)
        
                                clientid=Id
                                #print("clientid",clientid)
                        else:
                                Id=0
                                #profile=getProfile(Id)
                                cv2.putText(color_img,'0', (x,y+h),font, fontScale, fontColor)
                                cv2.putText(color_img,'UNKNOWN', (x,y+h+30),font, fontScale, fontColor)
                        cv2.imshow(filename, color_img)
                        if cv2.waitKey(1)==ord('q'):
                                return Id
                        else:
                                return 0
def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
        if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
                with open(model_path, 'rb') as f:
                        knn_clf = pickle.load(f)
        # Load image file and find face locations
        # X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
                return []
        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)
        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        # print(closest_distances)
        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

        
@app.route('/Detect',methods=['POST'])
def detection():
        print("detection")
        clientid=0
        buf_length = 10
        known_conf = 5
        buf = [[]] * buf_length
        i = 0
        process_this_frame = True
        cam = cv2.VideoCapture(0)
        face_names = []
        while True:
                ret, frame = cam.read()
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_frame = small_frame[:, :, ::-1]
                if process_this_frame:
                        predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
                process_this_frame = not process_this_frame
                for name, (top, right, bottom, left) in predictions:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        if name!="UNKNOWN" or name!="unknown" or name!="UnKnown":
                                face_names.append(name)
                                break
                        else:
                                continue
                buf[i] = face_names
                i = (i + 1) % buf_length
                # print(buf)
                # Display the resulting image
                cv2.imshow('In Camera', frame)
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        
        if len(face_names)>0:
                cam.release()
                cv2.destroyAllWindows()
                cmd="SELECT * FROM login WHERE name='"+str(face_names[0]+"'")
                conn.execute(cmd)
                cursor=conn.fetchall()
                print(cursor)
                results=[]
                for row in cursor:
                        print(row)
                        results.append(row[0])
                        results.append(row[1])
                        results.append(row[2])
                        results.append(row[3])
                        results.append(row[4])
                print(results)
                return render_template("login1.html",results=results)
                        
    

@app.route('/Addpin',methods=['POST'])
def addpin():
        print("addpin")
        cid=request.form['cid']
        pin1=request.form['pin']
        results=[]
        
        pin=str(pin1)+pingen.process()
        print(pin)
        print("len",len(str(pin)))
        while len(pin)<4:
                pin=str(pin)+pingen.process()
                print(pin)
                #render_template("login.html",results=results,pin=pin)
                
        results.append(cid)
        results.append(pin)
        print(results)
        return render_template("login2.html",results=results)
    
@app.route('/register',methods=['POST'])
def reg():
        name=request.form['name']
        cid=request.form['cid']
        pin=request.form['pin']
        email=request.form['emailid']
        mobile=request.form['mobile']
        cmd="SELECT * FROM login WHERE cid='"+cid+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Username Already Exists")
                return render_template("login.html",message="Client id Already Exists")
        else:
                print("insert")
                cmd="INSERT INTO login Values('"+str(cid)+"','"+str(name)+"','"+str(pin)+"','"+str(email)+"','"+str(mobile)+"','2000')"
                print(cmd)
                print("Inserted Successfully")
                conn.execute(cmd)
                mydb.commit()
                StartCamera(cid,name)
                knntrain.trainer()
                return render_template("login.html",message="Inserted SuccesFully")
@app.route('/changepin',methods=['POST'])
def upatepin():
        #name=request.form['name']
        cid=request.form['cid']
        pin=request.form['pin']
        cpin=request.form['cpin']
        #mobile=request.form['mobile']
        cmd="SELECT * FROM login WHERE cid='"+cid+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Username Already Exists")
                print("pin==",pin)
                print("cpin==",cpin)
                if int(pin)==int(cpin):
                        cmd="UPDATE login SET pin='"+str(pin)+"'WHERE cid='"+cid+"'"
                        print(cmd)
                        print("Inserted Successfully")
                        conn.execute(cmd)
                        mydb.commit()
                        return render_template("login.html",message="PIN Updated")
                else:
                        return render_template("forgetpass.html",message="Pin and Confirm Pin Mismatch")

                
        else:
                return render_template("forgetpass.html",message="Client id not Exist")

    
@app.route('/login1',methods=['POST'])
def log_in1():
        global otp
        otpvalue=request.form['otp']
        raco=request.form['rcid']
        amt=request.form['amt']
        cid=request.form['cid']
        pin=request.form['pin']
        cname=session.get('cname')
        bal=""

        print(otp)
        print(otpvalue)
        if otp==otpvalue:
                cmd="SELECT cid,pin,email,amt FROM login WHERE cid='"+cid+"' and pin='"+pin+"'"
                print(cmd)
                conn.execute(cmd)
                cursor=conn.fetchall()
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                        
                if(isRecordExist==1):
                        bal=row[3]
                        email=row[2]
                        if float(bal)<float(amt):
                                msg="Insuffecient Balance"
                                return render_template("login.html",message=msg)
                        else:
                                cmd="INSERT INTO transc(fromacc,toacc,amt,dat,acc_name) Values('"+str(cid)+"','"+str(raco)+"','"+str(amt)+"',now(),'"+str(cname)+"')"
                                print(cmd)
                                print("Inserted Successfully")
                                conn.execute(cmd)
                                mydb.commit()
                                cmd1="SELECT email,amt,name FROM login WHERE cid='"+raco+"'"
                                print(cmd1)
                                conn.execute(cmd1)
                                cursor=conn.fetchall()
                                isRecordExist=0
                                for row in cursor:
                                        isRecordExist=1
                                if(isRecordExist==1):
                                        bal1=row[1]
                                        remail=row[0]
                                        rname=row[2]
                                        total=float(bal1)+float(amt)
                                        total1=float(bal)-float(amt)
                                        cmd="update login set amt='"+str(total)+"' where cid='"+str(raco)+"'"
                                        print(cmd)
                                        print("Updated Balance Successfully")
                                        conn.execute(cmd)
                                        mydb.commit()
                                        cmd="update login set amt='"+str(total1)+"' where cid='"+str(cid)+"'"
                                        print(cmd)
                                        print("Updated Balance Successfully")
                                        conn.execute(cmd)
                                        mydb.commit()
                                        msg2="Dear "+str(rname)+", You have received Rs."+str(amt)+" from "+str(cid)+ "Your Current balance is:"+str(total)
                                        msg = EmailMessage()
                                        msg.set_content(str(msg2))
                                        msg['Subject'] = 'Tranasaction Details'
                                        msg['From'] = os.getenv('EMAIL_USER')
                                        msg['To'] = remail
                                        s = smtplib.SMTP('smtp.gmail.com', 587)
                                        s.starttls()
                                        s.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASSWORD'))
                                        s.send_message(msg)
                                        s.quit()
                                        msg1="Dear "+str(cname)+", Your cardless Tranasaction Done Successfully You have transfer Rs."+str(amt)+" to "+str(raco)+ "Your Current balance is:"+str(total1)
                                        msg = EmailMessage()
                                        msg.set_content(str(msg1))
                                        msg['Subject'] = 'Tranasaction Details'
                                        msg['From'] = os.getenv('EMAIL_USER')
                                        msg['To'] = email
                                        s = smtplib.SMTP('smtp.gmail.com', 587)
                                        s.starttls()
                                        s.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASSWORD'))
                                        s.send_message(msg)
                                        s.quit()
                                        session['logged_in'] = True
                                        return url_for('index')
                                else:
                                        return render_template("login.html",message="No Recevier Account ")
                else:
                        return render_template("login.html",message="No Sender Account ")
        else:
                return render_template("login.html",message="Transaction Failed Check OTP Value")
@app.route('/fpass1',methods=['POST'])
def fpass_in1():
        global otp
        otpvalue=request.form['otp']
        print(otp)
        print(otpvalue)
        if otp==otpvalue:
                results=[]
                #session['logged_in'] = True
                session['cid'] = request.form['cid']
                cid=request.form['cid']
                results.append(cid)
                return render_template("changepin.html",results=results)
        else:
                return render_template("forgetpass.html",message="Check OTP Value")
                

    
@app.route('/login',methods=['POST'])
def log_in():
        global otp
        #complete login if name is not an empty string or doesnt corss with any names currently used across sessions
        if request.form['cid'] != None and request.form['cid'] != "" and request.form['pin'] != None and request.form['pin'] != "":
                cid=request.form['cid']
                pin=request.form['pin']
                cmd="SELECT cid,name,pin,email FROM login WHERE cid='"+cid+"' and pin='"+pin+"'"
                print(cmd)
                conn.execute(cmd)
                cursor=conn.fetchall()
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                        
                if(isRecordExist==1):
                        mobile=row[3]
                        cname=row[1]
                        session['cname'] = cname
                        results=[]
                        results.append(cid)
                        results.append(pin)
                        print(otp)
                        otp=totp.now()
                        print(otp)
                        print("Email=",mobile)
                        msg = EmailMessage()
                        msg.set_content("Your OTP is : "+str(otp))
                        msg['Subject'] = 'OTP'
                        msg['From'] = "otpserver2024@gmail.com"
                        msg['To'] = mobile
                        s = smtplib.SMTP('smtp.gmail.com', 587)
                        s.starttls()
                        s.login("otpserver2024@gmail.com", "hjajxhrufxfmkwta")
                        s.send_message(msg)
                        s.quit()

                        
                        return render_template("otp.html",results=results)
                else:
                        return render_template("login.html",message="Check Clinet id and Pin Number")

        return redirect(url_for('index'))
@app.route('/fpass',methods=['POST'])
def fpass_in():
        global otp
        #complete login if name is not an empty string or doesnt corss with any names currently used across sessions
        if request.form['name'] != None and request.form['name'] != "" and request.form['cid'] != None and request.form['cid'] != "":
                cid=request.form['name']
                pin=request.form['cid']
                cmd="SELECT cid,name,pin,email FROM login WHERE cid='"+pin+"'"
                print(cmd)
                conn.execute(cmd)
                cursor=conn.fetchall()
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                        
                if(isRecordExist==1):
                        mobile=row[3]
                        uname=row[1]
                        results=[]
                        #results.append(cid)
                        results.append(pin)
                        print(otp)
                        otp=totp.now()
                        print(otp)
                        print("Email=",mobile)
                        msg = EmailMessage()
                        msg.set_content("Your OTP is : "+str(otp))
                        msg['Subject'] = 'OTP'
                        msg['From'] = "otpserver2024@gmail.com"
                        msg['To'] = mobile
                        s = smtplib.SMTP('smtp.gmail.com', 587)
                        s.starttls()
                        s.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASSWORD'))
                        s.send_message(msg)
                        s.quit()
                        
                        return render_template("otp1.html",results=results)
                else:
                        return render_template("forgetpass.html",message="Check Account number and name")

        return redirect(url_for('index'))
        
@app.route("/logout")
def log_out():
    session.clear()
    return redirect(url_for('index'))
    
@app.route("/AddClient")
def addClient():
    return render_template("dashboard.html")

@app.route("/TrainingPage")
def TrainingPage():
    return render_template("Training.html")
   
@app.route("/ViewClientPage")
def ViewClientPage():
    return render_template("dashboard1.html")
  
@app.route("/viewmytrans")
def viewmytrans():
        cname=session.get("cname")
        sql="select * from transc where acc_name='"+str(cname)+"'"
        conn.execute(sql)
        results = conn.fetchall()
        return render_template("dashboard2.html",result=results)  
        
    
# /////////socket io config ///////////////
#when message is recieved from the client    
@socketio.on('message')
def handleMessage(msg):
    print("Message recieved: " + msg)
 
# socket-io error handling
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass


  
  
if __name__ == '__main__':
    socketio.run(app,debug=True,host='127.0.0.1', port=4000)
