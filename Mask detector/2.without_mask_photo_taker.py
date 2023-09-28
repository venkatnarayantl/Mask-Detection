import cv2
import numpy as np
data = []
haar_data = cv2.CascadeClassifier('/home/pranav/Desktop/Python Project/Mask detector/data.xml')
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h,x:x+w, :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)< 400:
                data.append(face)
        cv2.imshow('result',img)
        #27 - ASCII of Escape
        if cv2.waitKey(2)==27 or len(data) >= 200:
            break
capture.release()
cv2.destroyAllWindows()
np.save('without_mask.npy',data)
