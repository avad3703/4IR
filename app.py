from flask import Flask
import plot_object_detection_saved_model as obj
import base64
import cv2 as cv
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_image():
    # products = obj.load_image(r"C:\Users\Sandali Thissera\Desktop\SyscoLabs\4IRProject\Training\models\research\decoded_image.jpg")
    return products