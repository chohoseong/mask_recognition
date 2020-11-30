import numpy as np
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
import glob

def image_generate(file, folder):
    np.random.seed(5)
    path = glob.glob(file)
    for k, f in enumerate(path):
        # 데이터셋 불러오기
        data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=5,
                                          width_shift_range=0.05,
                                          height_shift_range=0.05,
                                          shear_range=0.1,
                                          zoom_range=[0.9, 1.1],
                                          horizontal_flip=False,
                                          vertical_flip=False,
                                          fill_mode='nearest')

        img = load_img(f)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0

        # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
        for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=folder, save_prefix='tri',
                                       save_format='jpg'):
            i += 1
            if i > 99:
                break
        print('Colleting Samples Complete!!!')