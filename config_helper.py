import sys
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from cnn_models.cnn_pool import CNNPool
from cnn_models.cnn_dropout import CNNDropout
from cnn_models.ensemble import Ensemble

def retrieve_option_model(model_type):
    model_type = model_type.lower() if model_type else ""
    model = None
    if model_type == "hotdog":
        model = DeepDog(summarize=False)
    elif model_type == "cnn":
        model = CNN(summarize=False)
    elif model_type == "cnn_pool":
        model = CNNPool(summarize=False)
    elif model_type == "cnn_dropout":
        model = CNNDropout(summarize=False)
    elif model_type == "ensemble":
        model = Ensemble(summarize=False)
    else:
        model = DeepDog(summarize=False)

    return model
