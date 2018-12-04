import os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pprint import pprint
import json

def display_history(history_path):
    data = {}
    with open(history_path) as f:
        data = json.load(f)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    print("Maximum training accuracy: {}".format(max(data['acc'])))
    print("Maximum validation accuracy: {}".format(max(data['val_acc'])))

    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    print("Minimum training loss: {}".format(min(data["loss"])))
    print("Minimum validation loss: {}".format(min(data["val_loss"])))
