import os, sys
from keras.callbacks import ModelCheckpoint
from data_prep.prepare_dataset import load_dataset_from_file
import json

class Trainer:
    def __init__(self, model_object):
        self.model_object = model_object
        self.model = model_object.model
        self.epochs = 20
        self.checkpointer = model_object.checkpointer
        self.prepare_dataset()
        self.train()

    def prepare_dataset(self):
        training_input, training_output, validation_input, validation_output = \
            load_dataset_from_file()
        self.training_input = training_input
        self.training_output = training_output
        self.validation_input = validation_input
        self.validation_output = validation_output

    def train(self):
        history = self.model.fit(self.training_input, self.training_output,
                    epochs=self.epochs, validation_data=(self.validation_input, self.validation_output),
                    batch_size=48,
                    callbacks=[self.checkpointer], verbose=1, shuffle=True)
        model_name = self.model_object.name
        history_path = os.path.join("trainers", "history")
        history_path = os.path.join(history_path, model_name + ".json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)

if __name__ == '__main__':
    Trainer(DeepDog())
