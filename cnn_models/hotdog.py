from keras.layers import Input, Convolution2D, SeparableConvolution2D, \
                GlobalAveragePooling2D, GlobalMaxPooling2D, \
                Dense, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint

class DeepDog:
    def __init__(self, input_shape=(128, 128, 3), summarize=True):
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.channels = input_shape[2]
        self.input_shape = input_shape
        self.alpha = 1
        self.model = None
        self.checkpoint_path = 'models/hotdog.best.hdf5'
        self.checkpointer = ModelCheckpoint(filepath=self.checkpoint_path, verbose=1,
                                save_best_only=True)
        self.build_model()
        if summarize:
            self.model.summary()

    def build_model(self):
        model_input = Input(shape=self.input_shape)
        alpha = self.alpha
        activation_type = 'elu'

        # input format will usually be 128 or 2^7
        # strides of 2 halfs input shape
        x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same')(model_input)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        # why isn't separable convolution used as firstlayer
        x = SeparableConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = SeparableConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        for _ in range(5):
            x = self.apply_separable_layer(x, int(512 * alpha), activation_type)

        # GlobalAveragePooling susceptible to change
        x = GlobalAveragePooling2D()(x)
        # output activation type is subject to change
        out = Dense(1, activation='sigmoid')(x)

        self.model = Model(model_input, out, name='deepdog')
        # not sure what type of optimizer or loss function to use
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                                metrics=['accuracy'])

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

    # for this model type kernel size will be equal to 3
    def apply_separable_layer(self, x, filters, activation_type, strides=(1, 1)):
        x = SeparableConvolution2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)
        return x

if __name__ == '__main__':
    deep_dog = DeepDog()
