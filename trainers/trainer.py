from keras.callbacks import ModelCheckpoint
from data_prep.prepare_dataset import load_dataset

class Trainer:
    def __init__(self, model_object):
        self.model_object = model_object
        self.model = model_object.model
        self.epochs = 20
        self.checkpointer = ModelCheckpoint(filepath='models/detector.best.hdf5', verbose=1,
                                save_best_only=True)
        self.prepare_dataset()
        self.train()

    def prepare_dataset(self):
        training_input, training_output, validation_input, validation_output = \
            load_dataset()
        self.training_input = training_input
        self.training_output = training_output
        self.validation_input = validation_input
        self.validation_output = validation_output

    def train(self):
        self.model.fit(self.training_input, self.training_output,
                    epochs=self.epochs, validation_data=(self.validation_input, self.validation_output),
                    callbacks=[self.checkpointer], verbose=1, shuffle=True)

if __name__ == '__main__':
    Trainer(DeepDog())
