import os
import numpy as np
import Face_Recognition as f
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import shutil

def removefile(origin_folder, mask_folder):

    label = os.path.basename(origin_folder)

    shutil.rmtree(origin_folder)
    shutil.rmtree(mask_folder)

    mask_data=np.load('./model/face_embedding.npz')
    mask_trainX, mask_trainy = mask_data['arr_0'], mask_data['arr_1']
    print('mask: ',mask_trainX.shape,mask_trainy.shape)

    new_mask_TrainX = list()
    new_mask_Trainy = list()
    new_mask_TrainX.extend(mask_trainX)
    new_mask_Trainy.extend(mask_trainy)

    i=0
    while i != len(new_mask_Trainy):
        if new_mask_Trainy[i] == label:
            new_mask_Trainy.pop(i)
            new_mask_TrainX.pop(i)
        else: i=i+1

    new_mask_TrainX = np.asarray(new_mask_TrainX)
    new_mask_Trainy = np.asarray(new_mask_Trainy)
    print('mask: ',new_mask_TrainX.shape, new_mask_Trainy.shape)

    np.savez_compressed('./model/face_embedding.npz', new_mask_TrainX, new_mask_Trainy)

    f.trainX, f.trainy = new_mask_TrainX, new_mask_Trainy
    f.in_encoder = Normalizer(norm='l2')
    f.trainX = f.in_encoder.transform(f.trainX)

    # 목표 레이블 암호화
    f.out_encoder = LabelEncoder()
    f.out_encoder.fit(f.trainy)
    f.trainy = f.out_encoder.transform(f.trainy)

    f.svc_model = SVC(kernel='linear', probability=True)
    f.svc_model.fit(f.trainX, f.trainy)


    origin_data = np.load('model/origin_face_2.npz')
    origin_trainX, origin_trainy = origin_data['arr_0'], origin_data['arr_1']
    print('origin: ',origin_trainX.shape, origin_trainy.shape)

    new_origin_TrainX = list()
    new_origin_Trainy = list()
    new_origin_TrainX.extend(origin_trainX)
    new_origin_Trainy.extend(origin_trainy)

    i = 0
    while i != len(new_origin_Trainy):
        if new_origin_Trainy[i] == label:
            new_origin_Trainy.pop(i)
            new_origin_TrainX.pop(i)
            break
        else:
            i = i + 1

    new_origin_TrainX = np.asarray(new_origin_TrainX)
    new_origin_Trainy = np.asarray(new_origin_Trainy)
    print('origin: ', new_origin_TrainX.shape, new_origin_Trainy.shape)
    np.savez_compressed('model/origin_face_2.npz', new_origin_TrainX, new_origin_Trainy)

    f.known_face_encodings = new_origin_TrainX
    f.known_face_names = new_origin_Trainy