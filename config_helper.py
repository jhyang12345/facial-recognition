import sys
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN

def retrieve_option_model(model_type):
    model_type = model_type.lower()
    model = None
    if model_type == "hotdog":
        model = DeepDog()
    elif model_type == "cnn":
        model = CNN()
    else:
        model = DeepDog()
    return model
