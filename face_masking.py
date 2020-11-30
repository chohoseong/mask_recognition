import cv2,dlib
import numpy as np
from math import atan2, degrees

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img

def angle_between(p1, p2):
  xDiff = p2[0] - p1[0]
  yDiff = p2[1] - p1[1]
  return degrees(atan2(yDiff, xDiff))

def masking_img(img):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
    mask = cv2.imread('mask_image.png', cv2.IMREAD_UNCHANGED)

    ori = img.copy()
    faces = detector(img)

    face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    center_x = int(shape_2d[2][0] + (shape_2d[14][0] - shape_2d[2][0]) / 2)
    center_y = int((shape_2d[8][1] + (shape_2d[28][1] - shape_2d[8][1]) / 2))

    mask_size_x = int(shape_2d[14][0] - shape_2d[2][0])
    mask_size_y = int(shape_2d[8][1] - np.mean([shape_2d[27][1], shape_2d[28][1]], axis=0))

    angle = -angle_between(shape_2d[3], shape_2d[13])
    M = cv2.getRotationMatrix2D((mask.shape[1] / 2 , mask.shape[0] /2), angle, 1)
    rotated_mask = cv2.warpAffine(mask, M , (mask.shape[1], mask.shape[0]))

    result = overlay_transparent(ori, rotated_mask, center_x, center_y, overlay_size=(mask_size_x, mask_size_y))

    return result


