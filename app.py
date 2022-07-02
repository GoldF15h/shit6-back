from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

import myface

app = Flask(__name__)
CORS(app)

# route http posts to this method

@app.route('/api/test', methods=['POST'])
def test():
    file = request.files['image'].read() ## byte file
    npimg = np.fromstring(file, np.uint8)
    img,faces = myface.detectFaces(npimg)

    # resaultImage = myface.replaceAllFace(img,faces,0)
    # resaultImage = myface.shuffleAllFace(img,faces)
    # resaultImage = myface.replaceAllFaceWithOriginalPhoto(img,faces)
    
    # TODO : เปลี่ยน photo เป็นอันที่เลือก
    photo = file
    photo = np.fromstring(photo, np.uint8)
    photo = cv2.imdecode(photo,cv2.COLOR_BGR2RGB)
    resaultImage = myface.replaceAllFaceWithSelectedPhoto(img,faces,photo)

    # resaultImage = myface.blurImage(img)
    # resaultImage = myface.blurAllFace(img,faces)

    return jsonify({'status':str(resaultImage)})

@app.route("/")
def index():
  return "Hello World"