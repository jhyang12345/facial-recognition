from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

class Face_Net:
    def __init__(self):
        # self.init_model()
        self.load_model()
        self.threshold = 0.35

    def load_model(self):
        self.model = model_from_json(open("./facenet_model.json", "r").read())
        self.model.load_weights('./facenet_weights.h5')

    def verify_face(self, img1, img2):
        #produce 128-dimensional representation
        img1_representation = self.model.predict(preprocess_image('%s' % (img1)))[0,:]
        img2_representation = self.model.predict(preprocess_image('%s' % (img2)))[0,:]
        
        img1_representation = l2_normalize(img1_representation)
        img2_representation = l2_normalize(img2_representation)

        euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
        print("euclidean distance (l2 norm): ",euclidean_distance)
        if euclidean_distance < self.threshold:
            print("verified... they are same person")
        else:
            print("unverified! they are not same person!")
        return euclidean_distance, euclidean_distance < self.threshold

    def change_anchor(self, euclidean):
        return euclidean < self.threshold / 2

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

if __name__ == '__main__':
    facenet = Face_Net()
