from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_ann():
    """
    creates a NN with 2 hidden layers of 6 nodes each
    :return:
    """
    model = Sequential()
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1))

    return model



