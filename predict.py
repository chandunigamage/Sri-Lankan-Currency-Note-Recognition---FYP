import urllib.request
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf 
import cv2
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

img_width, img_height = 224, 224
#model_path = 'D:/Final FYP/Renewed/testImages/CashKeras-50-finetune_all-epoch-drpt-03-lr0000001.h5'
#model_path = 'D:/Final FYP/models/CashKeras-50-finetune_all-epoch-drpt-03-lr0000001.h5'
model_path = 'C:/Users/HP/Desktop/Chan/CashKeras-50-finetune_2-epoch-drpt-03-lr0001.h5'
#model_path = 'C:/Users/HP/Desktop/Chan/CashKeras-50-finetune_all-epoch-drpt-03-lr0000001.h5'
model = tf.keras.models.load_model(model_path,compile=False)


def camera():
    url = 'http://192.168.43.1:8080/shot.jpg'

    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imshow('temp', img)
        cv2.imwrite(filename='saved_img.jpg', img=img)
        q = cv2.waitKey(1)
        if q == ord("q"):
            break;


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("මුදල් නෝට්ටුවේ වටිනාකම :  රුපියල් පනහකි")
  elif answer == 1:
    print("මුදල් නෝට්ටුවේ වටිනාකම : රුපියල් පන්සියයක්")
  elif answer == 2:
    print("මුදල් නෝට්ටුවේ වටිනාකම : රුපියල් පන්දහසක්")
  elif answer == 3:
    print("මුදල් නෝට්ටුවේ වටිනාකම : රුපියල් සියය ")
  elif answer == 4:
    print("මුදල් නෝට්ටුවේ වටිනාකම : රුපියල් දහසක්")
  elif answer == 5:
    print("මුදල් නෝට්ටුවේ වටිනාකම : රුපියල් විස්සකි")

  return answer
  

if __name__ == '__main__': 
    camera()
    file = 'D:/Final FYP/saved_img.jpg'
    predict(file)
  

