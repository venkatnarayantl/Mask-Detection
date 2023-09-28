import cv2
haar_data = cv2.CascadeClassifier('/home/pranav/Desktop/Python Project/Face detector/data.xml')
img = cv2.imread('/home/pranav/Desktop/Python Project/Face detector/tom.png')
while True:
              faces = haar_data.detectMultiScale(img)
              for x,y,w,h in faces:
                  cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255),4)
              cv2.imshow('result',img)
              #27- ASCII of Escape
              if cv2.waitKey(2)== 27: 
                  break
cv2.destroyAllWindows()
