import sys
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from cnn_models.cnn_pool import CNNPool
from cnn_models.cnn_dropout import CNNDropout

def retrieve_option_model(model_type):
    model_type = model_type.lower()
    model = None
    if model_type == "hotdog":
        model = DeepDog()
    elif model_type == "cnn":
        model = CNN()
    elif model_type == "cnn_pool":
        model = CNNPool()
    elif model_type == "cnn_dropout":
        model = CNNDropout()
    else:
        model = DeepDog()

    return model
