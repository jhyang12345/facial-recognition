from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

class VGG_face:
    def __init__(self, epsilon=0.35):
        self.init_model()
        self.load_model()
        self.output_model = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)
        self.epsilon = epsilon
        pass

    def init_model(self):
        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(Convolution2D(4096, (7, 7), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(4096, (1, 1), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(2622, (1, 1)))
        self.model.add(Flatten())
        self.model.add(Activation('softmax'))

    def load_model(self):
        self.model.load_weights("./vgg_face_weights.h5")

    # resizing images to target size
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def verify_face(self, img1, img2):
        try:
            img1_representation = self.output_model.predict(self.preprocess_image(img1))[0,:]
            img2_representation = self.output_model.predict(self.preprocess_image(img2))[0,:]
        except Exception as e:
            print("Failed to open file!")
            return False

        cosine_similarity = self.findCosineSimilarity(img1_representation, img2_representation)
        euclidean_distance = self.findEuclideanDistance(img1_representation, img2_representation)
        print(cosine_similarity)
        return cosine_similarity, cosine_similarity < self.epsilon

    def change_anchor(self, cosine):
        return cosine < self.epsilon / 2


if __name__ == '__main__':
    face = VGG_face()
