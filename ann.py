from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.vis_utils import plot_model


def create_ann():
    """
    creates a NN with 2 hidden layers of 6 nodes each
    :return:
    """
    model = Sequential()
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    plot_model(model, show_shapes=True, show_layer_names=True)

    return model


def plot_ann(ann):
    plot_model(ann, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


