import numpy as np
import cv2
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')
with_mask.shape
without_mask.shape
with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)
without_mask.shape
with_mask.shape
X = np.r_[with_mask,without_mask]
X.shape
labels = np.zeros(X.shape[0])
labels[200:] = 1.0
names = {0: 'Mask', 1:'No Mask'}
#python - sklearn - scikit-learn
# svm - Support Vector Machine
#SVC - Support Vector Classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,labels,test_size=0.25 )
x_train.shape
# PCA - Princippal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
x_train= pca.fit_transform(x_train)
x_train[0]
x_train.shape
x_train,x_test,y_train,y_test = train_test_split(X,labels,test_size=0.25 )
svm = SVC()
svm.fit(x_train,y_train)
#x_test = pca.transform(x_test)
y_pred =svm.predict(x_test)
accuracy_score(y_test, y_pred)
import cv2
import numpy as np
haar_data = cv2.CascadeClassifier('/home/pranav/Desktop/Python Project/Mask detector/data.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h,x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            #face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244, 250, 0), 2)
            print(n)
        cv2.imshow('result',img)
        #27 - ASCII of Escape
        if cv2.waitKey(2)==27:
            break
capture.release()
cv2.destroyAllWindows()
