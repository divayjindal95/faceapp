import cv2
import dlib

class FaceDetect:

    def __init__(self, detector=None, image_path=None, video_path=None):


        self.detector=detector
        if self.detector=="HOG":
            self.hog(image_path,video_path)
        if self.detector=="HAAR":
            self.haar(image_path,video_path)


    def haar(self, image_path=None, video_path=None):

        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mouth.xml')

        if image_path!=None:
            image = cv2.imread(image_path)
            img=self.haarImage(image, face_cascade, eye_cascade, mouth_cascade)
            cv2.imshow('img', img)
            cv2.waitKey(0)

        if video_path!=None:
            video_capture = cv2.VideoCapture(video_path)
            print video_capture.isOpened()
            while (video_capture.isOpened()):
                ret, frame = video_capture.read()
                print frame
                if ret:
                    img=self.haarImage(frame, face_cascade, eye_cascade, mouth_cascade)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            video_capture.release()
            cv2.destroyAllWindows()

    def haarImage(self, image, face_cascade, eye_cascade, mouth_cascade):
        img = cv2.resize(image, (384, 288))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'visitor', (x, y), font, 0.5, (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            mouth = mouth_cascade.detectMultiScale(roi_gray)
            for (mx, my, mw, mh) in mouth:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        return img

    def hog(self, image_path=None, video_path=None):
        detector = dlib.get_frontal_face_detector()
        win = dlib.image_window()

        if image_path!=None:
            image = cv2.imread(image_path)
            dets=hogImage(image,detector)
            win.clear_overlay()
            win.set_image(image)
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()

        if video_path!=None:
            video_capture = cv2.VideoCapture(0)
            while (video_capture.isOpened()):
                ret, frame = video_capture.read()
                if ret:
                    dets=hogImage(frame, detector)
                    win.clear_overlay()
                    win.set_image(frame)
                    win.add_overlay(dets)

    def hogImage(self, image, detector):
        dets = detector(image, 1)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.

        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))

        return dets
