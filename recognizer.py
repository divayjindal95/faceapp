from dic import dic1
from dic import dic2
import cv2
import  os
import numpy as np
from PIL import Image
import dlib
class Recognizer:

    def __init__(self,):
        self.recognizer = cv2.createLBPHFaceRecognizer()

    def train(self):
        path = './aligned_faces'
        image_paths = os.listdir(path)
        images = []
        labels = []
        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(path + '/' + image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            label = dic1[image_path.split()[0]]
            images.append(image)
            labels.append(int(label))

        self.recognizer.train(images, np.array(labels))
        self.recognizer.save('trained.xml')

    def test(self, alignedFace):
        gray = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
        nbr_predicted, conf = self.recognizer.predict(gray)
        print " predicted as {} with confidence {}".format(dic2[nbr_predicted], conf)
        return  nbr_predicted


if __name__=='__main__':
    rc=Recognizer()
    rc.train()

    rc.test(cv2.imread('./test_faces/2.jpg'))

