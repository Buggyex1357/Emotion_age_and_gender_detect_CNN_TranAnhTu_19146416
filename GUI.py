from tkinter import * 
import tkinter
from tkinter.font import BOLD, ITALIC
from tkinter.ttk import Combobox 
from tkinter import messagebox
import PIL.Image, PIL.ImageTk
import cv2

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

face_classifier=cv2.CascadeClassifier(r'C:\Users\P50\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
emotion_model = load_model(r'C:\Users\P50\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\emotion2.h5')
age_model = load_model(r'C:\Users\P50\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\age.h5')
gender_model = load_model(r'C:\Users\P50\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\gender_p6.h5')

class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
gender_labels = ['Male', 'Female']

window = Tk()
window.title("Multi-detect")
window.geometry("1000x800")

# Thêm label
univer = Label(window, text = 'TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT', fg = 'red', font = ("Arial", 20,BOLD))
univer.place(x=235,y =0)

ckm = Label(window, text = 'KHOA CƠ KHÍ CHẾ TẠO MÁY', fg = 'blue', font = ("Arial", 20,BOLD))
ckm.place(x=300,y =35)

project = Label(window, text = 'PROJECT CUỐI KÌ MÔN AI:', fg = 'brown', font = ("Arial", 25,BOLD, UNDERLINE))
project.place(x=280,y =85)

title = Label(window, text = 'Dự đoán cảm xúc, tuổi tác và giới tính thời gian thực', fg = 'brown', font = ("Arial", 25,BOLD, ITALIC))
title.place(x=100,y =130)

name = Label(window, text = 'SVTH: Trần Anh Tú', fg = 'black', font = ("Arial", 14,BOLD))
name.place(x=750,y =185)

mssv = Label(window, text = 'MSSV: 19146416', fg = 'black', font = ("Arial", 14,BOLD))
mssv.place(x=750,y =210)

frame1 = cv2.imread('CKM.png')
frame1 = cv2.resize(frame1,(110,110))
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
 # Convert hanh image TK
photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
# Show
logo_ute = Label(window, image= photo1)
logo_ute.place(x=0,y =0)

frame2 = cv2.imread('CKM1.png')
frame2 = cv2.resize(frame2,(110,110))
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
 # Convert hanh image TK
photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
# Show
logo_ute = Label(window, image= photo2)
logo_ute.place(x=885,y =0)

video = cv2.VideoCapture(0)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)*1.2 
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)*1.08

canvas = Canvas(window, width = canvas_w, height= canvas_h , bg= "gray")
canvas.place(x = 70, y =240)
bw = 0
detect = 0
onoff = 0

def handleBW():
    global bw
    bw = 1 - bw

def detecOnOff():
    global detect
    detect = 1 - detect

def onOff():
    global onoff
    onoff = 1 - onoff
    
button = Button(window,text = "Camera",font = ("Arial",8,BOLD) ,command=handleBW, width=14, height=2, background="yellow" ,activebackground = "green")
button.place(x = 860,y = 550)

button1 = Button(window,text = "Detect",font = ("Arial",8,BOLD) ,command=detecOnOff, width=14, height=2, background="yellow" ,activebackground = "green")
button1.place(x = 860,y = 600)

button2 = Button(window,text = "Quit",font = ("Arial",8,BOLD) , command=onOff, width=14, height=2, background="red" ,activebackground = "green")
button2.place(x = 860,y = 700)

def update_frame():
    global canvas, photo, bw, count
    
    if bw == 1:
        # Doc tu camera
        ret, frame = video.read()
        # Ressize
        frame = cv2.resize(frame, dsize= (800,515))
        # Chuyen he mau
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert hanh image TK
        # Show
        
        labels=[]
        
        faces=face_classifier.detectMultiScale(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        
        if detect == 1:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                #Lay hinh danh de du doan
                roi=roi_gray.astype('float')/255.0  #Scale
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)

                #Emotion
                preds=emotion_model.predict(roi)[0]  
                label=class_labels[preds.argmax()] 
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
                #Gender
                roi_color=frame[y:y+h,x:x+w]
                roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
                gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
                gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
                gender_label=gender_labels[gender_predict[0]] 
                gender_label_position=(x,y+h+50)
                cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
                #Age
                age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
                age = round(age_predict[0,0])
                age_label_position=(x+h,y+h)
                cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
    if onoff ==1:
        video.release()
        window.destroy()    
    window.after(15, update_frame)

update_frame()

window.mainloop()
