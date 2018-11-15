from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

class VGG_face:
    def __init__(self):
        self.init_model()
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
    
