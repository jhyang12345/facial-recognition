from keras.layers import Input, Convolution2D, SeparableConvolution2D, \
                GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, \
                Dense, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint

class CNN:
    def __init__(self, input_shape=(128, 128, 3), summarize=True):
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.channels = input_shape[2]
        self.input_shape = input_shape
        self.alpha = 1
        self.name = "cnn"
        self.model = None
        self.checkpoint_path = 'models/regular_cnn.best.hdf5'
        self.checkpointer = ModelCheckpoint(filepath=self.checkpoint_path, verbose=1,
                                save_best_only=True)
        self.build_model()
        if summarize: self.model.summary()

    def build_model(self):
        model_input = Input(shape=self.input_shape)
        alpha = self.alpha
        activation_type = 'elu'

        # input format will usually be 128 or 2^7
        # strides of 2 halfs input shape
        # usually kernel sizes are in odd numbers
        # kernel strides alternate between 1 and 2 so that we don't miss out
        x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same')(model_input)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(64 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        # GlobalAveragePooling susceptible to change
        x = GlobalAveragePooling2D()(x)
        # output activation type is subject to change
        out = Dense(1, activation='sigmoid')(x)

        self.model = Model(model_input, out, name='cnn')
        # not sure what type of optimizer or loss function to use
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                                metrics=['accuracy'])

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

if __name__ == '__main__':
    cnn = CNN()
