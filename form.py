import sys, cv2, os
from Face_Recognition import my_face_recognition
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from face_masking import masking_img
from face_crop import mask_cropping
from image_generator import image_generate
from add_embedding import add_person
from remove import removefile
from img_resize import resize_img

projectUI = './UI/project.ui'
registerUI = './UI/register.ui'
removeUI = './UI/remove.ui'

cap = cv2.VideoCapture(0)
class MainDialog(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi(projectUI, self)
        self.show()
        self.setWindowTitle('Main')
        #self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.ViewCam)
        self.cam_button.clicked.connect(self.CamControl)
        self.register_button.clicked.connect(self.register)
        self.remove_button.clicked.connect(self.remove)

    def CamControl(self):
        if not self.timer.isActive():
            # start timer
            #self.cap = cv2.VideoCapture(0)
            self.timer.start(20)
            # update control_bt text
            self.cam_button.setText("cam_off")
            # if timer is started
        else:
            self.timer.stop()
            #self.cap.release()
            self.cam_label.setPixmap(QPixmap.fromImage(QImage()))
            self.cam_button.setText("cam_on")


    def ViewCam(self):
        ret, img = cap.read()
        tmp = img.copy()
        result = my_face_recognition(tmp)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        qt_image = QImage(result, result.shape[1], result.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image)
        self.cam_label.setPixmap(pix)


    def register(self):
        self.timer.stop()
        self.cam_label.setPixmap(QPixmap.fromImage(QImage()))
        self.cam_button.setText("cam_on")
        registerDialog(self)


    def remove(self):
        self.timer.stop()
        self.cam_label.setPixmap(QPixmap.fromImage(QImage()))
        self.cam_button.setText("cam_on")
        removeDialog(self)


class registerDialog(QDialog):

    def __init__(self, parent):
        super(registerDialog, self).__init__(parent)
        uic.loadUi(registerUI,self)

        self.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.registCam)
        self.timer.start(20)

        self.regist_Button.clicked.connect(self.regist)
        self.cancel_Button.clicked.connect(self.close)

    def registCam(self):
        global img
        ret, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_image = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image)
        self.regist_label.setPixmap(pix)

    def regist(self):
        self.timer.stop()
        pattern = self.Name_Edit.text()

        try:
            mask_img = masking_img(img)
            result = mask_cropping(mask_img)

            msg_box = QMessageBox(self)
            x = msg_box.question(self, ' ', 'Would you like to register?')

            if x == QMessageBox.No:
                self.timer.start(20)
                return

            origin_folder = './face/' + 'train_origin/' + pattern
            if not os.path.exists(origin_folder):
                os.mkdir(origin_folder)
            else:
                x = msg_box.question(self,' ', 'File already exists. Would you like to overwrite it?')
                if x == QMessageBox.No:
                    self.timer.start(20)
                    return


            mask_folder = './face/' + 'train_mask/' + pattern

            if not os.path.exists(mask_folder):
                os.mkdir(mask_folder)

            origin_file = origin_folder + '/' + pattern + '.jpg'
            mask_file = mask_folder + '/' + pattern + '_mask.jpg'

            cv2.imwrite(origin_file, img)
            cv2.imwrite(mask_file,result)
            image_generate(mask_file, mask_folder)

            add_person(origin_folder,mask_folder)

            msg_box.question(self, ' ', 'complete', QMessageBox.Yes)
            self.timer.start(20)

        except:
            msg_box = QMessageBox(self)
            msg_box.question(self, ' ', 'try again', QMessageBox.Yes)
            self.timer.start(20)


    def keyReleaseEvent(self, *args, **kwargs):
        if self.Name_Edit.text():
            self.regist_Button.setEnabled(True)
        else:
            self.regist_Button.setEnabled(False)


class removeDialog(QDialog):

    def __init__(self, parent):
        super(removeDialog, self).__init__(parent)
        uic.loadUi(removeUI, self)

        self.show()
        self.load_Button.clicked.connect(self.load)
        self.remove_Button.clicked.connect(self.remove)
        self.cancel_Button.clicked.connect(self.close)

    def load(self):
        pattern = self.Name_Edit.text()
        origin_file = './face/train_origin/' + pattern + '/' + pattern + '.jpg'

        try:
            img = cv2.imread(origin_file)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = resize_img(image)
            qt_image = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_image)
            self.remove_label.setPixmap(pix)
        except:
            msg_box = QMessageBox(self)
            msg_box.question(self, ' ', 'no person', QMessageBox.Yes)


    def remove(self):
        pattern = self.Name_Edit.text()
        origin_folder = './face/train_origin/' + pattern
        mask_folder = './face/train_mask/' +pattern
        try:
            removefile(origin_folder,mask_folder)
            msg_box = QMessageBox(self)
            msg_box.question(self, ' ', 'complete', QMessageBox.Yes)
        except:
            msg_box = QMessageBox(self)
            msg_box.question(self, ' ', 'no person', QMessageBox.Yes)

    def keyReleaseEvent(self, *args, **kwargs):
        if self.Name_Edit.text():
            self.load_Button.setEnabled(True)
        else:
            self.load_Button.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_dialog = MainDialog()
    main_dialog.show()
    app.exec_()
