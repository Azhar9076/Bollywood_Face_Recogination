import os
import cv2 as cv
import numpy as np 

people = ['Esha Gupta', 'Alia Bhatt','Bipasha Basu','Huma Qureshi','Aishwarya Rai Bachchan','Anushka Sharma','Deepika Padukone','Genelia D_Souza','Bhumi Pednekar','Disha Patani','Aftab Shivdasani', 'Ajay Devgn', 'Abhishek Bachchan', 'Aditya Roy Kapur', 'Aamir Khan']
DIR = r'C:\Users\azahar patel\Downloads\bollywood 1'

features = []
labels = []

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)

            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            for (x,y,w,h) in face_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training is train.....')


# print(f'Length of features: {len(features)}')
# print(f'Length of labels: {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer
face_recognizer.train(features, labels)

# Save model and data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)