from detection import FaceDetect
import dlib
import cv2,os
from align_dlib import *
predictor_model = "shape_predictor_68_face_landmarks.dat"

class Transform:

    def __init__(self,image, dets, name):
        self.face_pose_predictor = dlib.shape_predictor(predictor_model)
        self.face_aligner = AlignDlib(predictor_model)
        self.alignAndSave(image,dets, name)

    def alignAndSave(self,image,detected_faces,name):
        for i, face_rect in enumerate(detected_faces):
            pose_landmarks = self.face_pose_predictor(image, face_rect)
            alignedFace = self.face_aligner.align(534, image, face_rect, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite("./aligned_faces/{}.jpg".format(name), alignedFace)

if __name__=="__main__":
    print "class"
    #fc=FaceDetect(detector='HAAR',image_path='../download.jpg')
    image_paths = os.listdir("./faces")
    i =0
    print "hello"
    for image_path in image_paths:
	print image_path
        fc = FaceDetect()
        image=cv2.imread(os.path.join('./faces',image_path))
        dets=fc.hogImage(image, dlib.get_frontal_face_detector())
        tf=Transform(image=image,dets=dets,name='arujit'+str(i))
        i +=1
