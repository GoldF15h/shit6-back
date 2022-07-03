from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

import myface

app = Flask(__name__)
CORS(app)

photo = cv2.imread('stupid_logo.jpeg')
# photo = cv2.imdecode(photo,cv2.COLOR_BGR2RGB)

# route http posts to this method

@app.route('/api/preprocess', methods=['POST'])
def test():
    file = request.files['image'].read() ## byte file
    faceSelected = int(request.form.get('faceSelected', -1))
    templateSelected = int(request.form.get('template', -1))

    npimg = np.fromstring(file, np.uint8)
    img, faces = myface.detectFaces(npimg)
    resultImage = ''


    if(templateSelected == -1):
      resultImage = myface.originalImage(img, faces)
    elif(templateSelected == 0):
      resultImage = myface.replaceAllFace(img, faces, faceSelected)
    elif(templateSelected == 1):
      resultImage = myface.shuffleAllFace(img,faces)
    elif(templateSelected == 2):
      resultImage = myface.replaceAllFaceWithOriginalPhoto(img,faces)
    elif(templateSelected == 3):
      resultImage = myface.replaceAllFaceWithSelectedPhoto(img,faces,photo)
    elif(templateSelected == 4):
      resultImage = myface.blurImage(img)
    elif(templateSelected == 5):
      resultImage = myface.blurAllFace(img,faces)

    return jsonify({'faces': faces.tolist(), 'size':{'w': img.shape[1], 'h': img.shape[0]},'image':str(resultImage)})

@app.route("/")
def index():
  return "Hello World"