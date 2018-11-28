from keras.layers import Input, Convolution2D, SeparableConvolution2D, \
                GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, \
                Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint

class CNNDropout:
    def __init__(self, input_shape=(128, 128, 3)):
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.channels = input_shape[2]
        self.input_shape = input_shape
        self.alpha = 1
        self.model = None
        self.checkpoint_path = 'models/cnn_dropout.best.hdf5'
        self.checkpointer = ModelCheckpoint(filepath=self.checkpoint_path, verbose=1,
                                save_best_only=True)
        self.build_model()
        self.model.summary()

    def build_model(self):
        model_input = Input(shape=self.input_shape)
        alpha = self.alpha
        activation_type = 'relu'
        # applying dropout factor to prevent overfitting
        dropout_factor = 0.4

        # input format will usually be 128 or 2^7
        # strides of 2 halfs input shape
        # usually kernel sizes are in odd numbers
        # kernel strides alternate between 1 and 2 so that we don't miss out
        x = Convolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same')(model_input)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(64 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        # kernel size of 3  halfs the input dimensions
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

        x = Dropout(dropout_factor)(x)

        x = Convolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

        x = Dropout(dropout_factor)(x)

        x = Convolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

        x = Dropout(dropout_factor)(x)

        x = Convolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        x = Convolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation_type)(x)

        # basically flattens a dimension
        x = GlobalMaxPooling2D()(x)

        # maybe add another dense layer in between
        out = Dense(1, activation='sigmoid')(x)

        self.model = Model(model_input, out, name='cnn_dropout')
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                                metrics=['accuracy'])

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

if __name__ == '__main__':
    CNNPool()
