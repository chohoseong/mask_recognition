from keras.models import load_model
import numpy as np
import cv2, os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import face_recognition

mask_model = load_model('./model/mask_classify.h5', compile=False)
facenet_model = load_model('./model/facenet_keras.h5')
mask_cascade = cv2.CascadeClassifier('./model/cascade.xml')

data = np.load('./model/face_embedding.npz')
trainX, trainy = data['arr_0'], data['arr_1']
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# 목표 레이블 암호화
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(trainX, trainy)

o_data=np.load('model/origin_face_2.npz')
known_face_encodings,known_face_names = o_data['arr_0'], o_data['arr_1']


def get_embedding(model, face_pixels):
    # 픽셀 값의 척도
    face_pixels = face_pixels.astype('int32')
    # 채널 간 픽셀값 표준화(전역에 걸쳐)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # 얼굴을 하나의 샘플로 변환
    samples = np.expand_dims(face_pixels, axis=0)
    # 임베딩을 갖기 위한 예측 생성
    yhat = model.predict(samples)
    return yhat[0]

def my_face_recognition(img):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160,160))
    data=np.asarray(image)

    X = []
    X.append(data)
    X=np.array(X)
    X=X.astype(float)/255

    prediction = mask_model.predict(X)

    if prediction >= 0.5:

        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            print(face_distances[best_match_index])

            if face_distances[best_match_index] >= 0.35:
                name = "Unknown"
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img, name, (left+6, top-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)

        cv2.putText(img, "NO_mask", (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = mask_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi,(160,160))
            face_pixels = np.asarray(roi)

            ip = []
            embedding = get_embedding(facenet_model, face_pixels)
            ip.append(embedding)

            testX = in_encoder.transform(ip)

            samples = np.expand_dims(testX[0], axis=0)
            yhat_class = svc_model.predict(samples)
            yhat_prob = svc_model.predict_proba(samples)

            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)


            if class_probability > 25:
                cv2.putText(img, predict_names[0], (x+6, y-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
                cv2.putText(img, '%.3f'%class_probability,(x, y+h+10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
            else:
                cv2.putText(img, 'unknown', (x+6, y-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)

        cv2.putText(img, "mask", (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    return img