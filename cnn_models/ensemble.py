from keras.layers import Input, Average
from keras.models import Model
from cnn_models.ensemble_models.cnn_dropout import CNNDropout
from cnn_models.ensemble_models.cnn_pool import CNNPool
from cnn_models.ensemble_models.cnn import CNN
from cnn_models.ensemble_models.hotdog import DeepDog

# ensemble model is not for training! only for evaluation!!!
class Ensemble:
    def __init__(self, input_shape=(128, 128, 3), summarize=False):
        self.image_width = input_shape[0]
        self.image_height = input_shape[1]
        self.channels = input_shape[2]
        self.input_shape = input_shape
        self.alpha = 1
        self.name = "ensemble"
        self.model = None

        self.build_model()
        self.compile_model()

    def build_model(self):
        model_input = Input(shape=self.input_shape)

        model_objects = []
        model_objects.append(DeepDog(model_input, summarize=False))
        model_objects.append(CNN(model_input, summarize=False))
        model_objects.append(CNNPool(model_input, summarize=False))
        model_objects.append(CNNDropout(model_input, summarize=False))

        # individual models should already be loaded
        for model in model_objects:
            model.load_model()

        outputs = [model.model.outputs[0] for model in model_objects]
        y = Average()(outputs)

        ensemble_model = Model(model_input, y, name=self.name)
        self.model = ensemble_model

    def load_model(self):
        print("Model already loaded!")

    def compile_model(self):
        # not sure what type of optimizer or loss function to use
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                                metrics=['accuracy'])

    def evaluate(self, input_data):
        output_data = model.predict(input_data, batch_size = 32)
        return output_data

if __name__ == '__main__':
    model = Ensemble()
