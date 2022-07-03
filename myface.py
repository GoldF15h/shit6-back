from random import random
import cv2
from PIL import Image
import base64
import os , io , sys
import numpy as np
import random

cascPath = 'detect/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

def detectFaces (npimg) :
    img = cv2.imdecode(npimg,cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return img, faces

def getMasterFace (img, faces, selectedFaceIndex) : 
    face = faces[selectedFaceIndex]
    (y, x, w, h) = face
    face_master = img[x:x+w, y:y+h] 
    return face_master

def saveImage (img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    return base64.b64encode(rawBytes.read())

def originalImage (img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 8)
    return saveImage(img)

def replaceAllFace (img, faces, selectedFaceIndex) :
    face_master = getMasterFace (img, faces, selectedFaceIndex)
    for (y, x, w, h) in faces:
      img[x:x+w, y:y+h] = cv2.resize(face_master, (w, h), interpolation = cv2.INTER_AREA)
    return saveImage(img)

def shuffleAllFace (img, faces) :
    randomFace = faces.copy()
    np.random.shuffle(randomFace)
    for selectedFaceIndex in range(len(faces)) :
        face_master = getMasterFace (img, randomFace, selectedFaceIndex)
        (y, x, w, h) = faces[selectedFaceIndex]
        img[x:x+w, y:y+h] = cv2.resize(face_master, (w, h), interpolation = cv2.INTER_AREA)
    return saveImage(img)

def blurAllFace (img, faces ) :
    for selectedFaceIndex in range(len(faces)) :
        face_master = getMasterFace (img, faces, selectedFaceIndex)
        cur_w,cur_h,c = face_master.shape
        blurValue = 0.1
        face_master = cv2.blur(face_master, (int(cur_w*blurValue),int(cur_h*blurValue)))
        curFace = faces[selectedFaceIndex]
        (y, x, w, h) = curFace
        blurValue = 0.1
        curFace = cv2.blur(face_master, (int(cur_w*blurValue),int(cur_h*blurValue)))
        img[x:x+w, y:y+h] = cv2.resize(curFace, (w, h), interpolation = cv2.INTER_AREA)
    return saveImage(img)

def blurImage (img) :
    w,h,c = img.shape
    blurValue = 0.01
    img = cv2.blur(img, (int(w*blurValue),int(h*blurValue)))
    return saveImage(img)

def replaceAllFaceWithOriginalPhoto (img, faces) :
    for (y, x, w, h) in faces:
      img[x:x+w, y:y+h] = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    return saveImage(img)

def replaceAllFaceWithSelectedPhoto (img, faces, photo) :
    for (y, x, w, h) in faces:
      img[x:x+w, y:y+h] = cv2.resize(photo, (w, h), interpolation = cv2.INTER_AREA)
    return saveImage(img)

