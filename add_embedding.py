import Face_Recognition as f
from PIL import Image
import os, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import face_recognition

def load_faces(directory,label):
	directory += '/'
	faces = list()
	Y = list()
	# 파일 열거
	for filename in os.listdir(directory):
		# 경로
		path = directory + filename
		# 얼굴 추출
		image = Image.open(path)
		image = image.convert('RGB')
		image = image.resize((160,160))
		face = np.asarray(image)
		# 저장
		faces.append(face)
		Y.append(label)
	return np.asarray(faces), np.asarray(Y)


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

def add_person(origin_folder,mask_folder):

	label = os.path.basename(mask_folder)
	print(label)

	origin_data = np.load('model/origin_face_2.npz')
	origin_trainX, origin_trainy = origin_data['arr_0'], origin_data['arr_1']
	print('origin: ', origin_trainX.shape, origin_trainy.shape)

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

	for file in os.listdir(origin_folder+'/'):
		print(file)
		image = face_recognition.load_image_file(origin_folder + '/' + file)
		encoding = face_recognition.face_encodings(image)[0]
		new_origin_TrainX.append(encoding)
		new_origin_Trainy.append(file.replace('.jpg', ''))

	new_origin_TrainX = np.asarray(new_origin_TrainX)
	new_origin_Trainy = np.asarray(new_origin_Trainy)
	print('origin: ', new_origin_TrainX.shape, new_origin_Trainy.shape)

	np.savez_compressed('model/origin_face_2.npz', new_origin_TrainX, new_origin_Trainy)

	f.known_face_encodings = new_origin_TrainX
	f.known_face_names = new_origin_Trainy


	mask_tx, mask_ty = load_faces(mask_folder,label)
	print(mask_ty[0])

	mask_data=np.load('./model/face_embedding.npz')
	mask_trainX, mask_trainy = mask_data['arr_0'], mask_data['arr_1']
	print('mask: ',mask_trainX.shape, mask_trainy.shape)

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

	new_mask_Trainy.extend(mask_ty)
	for face_pixels in mask_tx:
		embedding = get_embedding(f.facenet_model, face_pixels)
		new_mask_TrainX.append(embedding)
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





