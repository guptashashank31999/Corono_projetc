from django.db.models import Avg, Count
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import os
import requests
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.filedialog
import cv2
from skimage import exposure
import pickle
import sys
import imutils
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from scipy import linalg
import numpy
import argparse


from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# Create your views here.
from user.models import RegisterModel, DetectionModel


def login(request):

    if request.method=="POST":
        usid=request.POST.get('username')
        pswd = request.POST.get('password')
        try:
            check = RegisterModel.objects.get(userid=usid,password=pswd)
            request.session['userd_id']=check.id
            return redirect('mydetails')
        except:
            pass
    return render(request,'user/login.html')




def register(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        userid = request.POST.get('userid')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        email = request.POST.get('email')
        gender = request.POST.get('gender')

        if RegisterModel.objects.filter(userid=userid).exists():
            return HttpResponse("Username already taken")
        if RegisterModel.objects.create(firstname=firstname, lastname=lastname, userid=userid, password=password,
                                        phoneno=phoneno, email=email, gender=gender):
            return redirect('login')
    return render(request, 'user/register.html')


def mydetails(request):
    name = request.session['userd_id']
    ted = RegisterModel.objects.get(id=name)

    return render(request, 'user/mydetails.html',{'objects':ted})

def wellcome(request):
    name = request.session['userd_id']
    ted = RegisterModel.objects.get(id=name)

    return render(request, 'user/wellcome.html',{'objects':ted})

def updata_details(request):
    userd_id = request.session['userd_id']
    obj = RegisterModel.objects.get(id=userd_id)
    if request.method == "POST":
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        userid = request.POST.get('userid')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        email = request.POST.get('email')
        gender = request.POST.get('gender')


        obj = get_object_or_404(RegisterModel, id=userd_id)
        obj.firstname = firstname
        obj.lastname = lastname
        obj.userid = userid
        obj.password = password
        obj.phoneno = phoneno
        obj.email = email
        obj.gender = gender


        obj.save(update_fields=["firstname","lastname",  "userid", "password", "phoneno","email","gender"])
        return redirect('mydetails')

    return render(request, 'user/updata_details.html',{'form':obj})




def covid(request):
    ind=''
    def select_image1():
        # grab a reference to the image panels
        global panelA, panelB

        # open a file chooser dialog and allow the user to select an input
        # image
        path = filedialog.askopenfilename()
        print(path)
        # Initialising the CNN
        classifier = Sequential()

        # Convolution
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

        # Pooling
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a second convolutional layer
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattening
        classifier.add(Flatten())

        # Full connection
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=1, activation='sigmoid'))

        # Compiling the CNN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the CNN to the images

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory('dataset',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')

        test_set = test_datagen.flow_from_directory('dataset',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

        classifier.fit_generator(training_set,
                                 steps_per_epoch=200,
                                 epochs=3,
                                 validation_data=test_set,
                                 validation_steps=50)

        ##Prediction Part

        img_pred = image.load_img(path, target_size=(64, 64))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        rslt = classifier.predict(img_pred)

        ind = training_set.class_indices
        print()

        if rslt[0][0] == 1:
            prediction = "Normal"
        else:
            prediction = "COVID"

        ##Save model to json
        import os
        from keras.models import model_from_json

        clssf = classifier.to_json()
        with open("Covid.json", "w") as json_file:
            json_file.write(clssf)
        classifier.save_weights("Covid.h5")
        print("model saved to disk....")
        print(prediction)

        img_src = path
        img = cv2.imread(img_src)

        img_pred = image.load_img(path, target_size=(64, 64))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        rslt = classifier.predict(img_pred)

        ind = training_set.class_indices

        if rslt[0][0] == 1:
            prediction = "Normal"
        else:
            prediction = "COVID"

        target_width = 50
        target_height = 50
        target_size = (target_width, target_height)

        img = cv2.resize(img, target_size)
        img = img.reshape(1, target_width, target_height, 3)

        fig, ax = plt.subplots()
        fig.suptitle(prediction, fontsize=12)

        np_img = mpimg.imread(img_src)
        plt.imshow(np_img)

        plt.show()
        name = request.session['userd_id']
        userObj = RegisterModel.objects.get(id=name)
        DetectionModel.objects.create(covid_Userd=userObj, analysisvalue=prediction, image=path)

    # initialize the window toolkit along with the two image panels
    root = Tk()
    panelA = None
    panelB = None

    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    btn = Button(root, text="Select Your Xray image", command=select_image1)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    root.mainloop()

    Vobj = DetectionModel.objects.all()
    return render(request, 'user/covid.html', {'v': Vobj})






def charts(request,chart_type):
    chart = DetectionModel.objects.values('analysisvalue').annotate(dcount=Count('analysisvalue'))
    return render(request,'user/charts.html',{'chart_type':chart_type,'form':chart})



