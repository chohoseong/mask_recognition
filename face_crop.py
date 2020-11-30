import cv2

def mask_cropping(img):
    face_cascade=cv2.CascadeClassifier('./model/cascade.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3,10)
    face=[]
    face.append(faces[0])

    for (x, y, w, h) in face:
        roi_color = img[y:y + h, x:x + w]

    return roi_color
