from keras.layers import Input, Convolution2D, SeparableConvolution2D, \
                GlobalAveragePooling2D, GlobalMaxPooling2D,
                Dense, Activation, BatchNormalization

class DeepDog:
    def __init__(self, input_shape=(128, 128, 3)):
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.channels = input_shape[2]
        self.input_shape = input_shape
        self.alpha = 1
        self.build_model()

    def build_model(self):
        model_input = Input(shape=self.input_shape)
        alpha = self.alpha
        activation_type = 'elu'

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

        x = GlobalAveragePooling2D()(x)
        # output activation type is subject to change
        out = Dense(1, activation='sigmoid')(x)

        model = Model(model_input, out, name='deepdog')

    # for this model type kernel size will be equal to 3
    def apply_separable_layer(x, filters, activation_type, strides=(1, 1)):
        x = SeparableConvolution2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)
        return x
