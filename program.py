import numpy as np
import pandas as pd
import data_preprocessing as dp
import ann
from sklearn.metrics import r2_score
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':

    # data preprocessing
    data = pd.read_excel("data/Folds5x2_pp.xlsx")
    data_preprocessor = dp.DataPreprocessing(data)
    data_preprocessor.split_data(test_size=0.2)
    x_train, x_test, y_train, y_test = data_preprocessor.get_processed_data()
    x_train_scaled, x_test_scaled = data_preprocessor.rescaled_data()

    # model creation
    model = ann.create_ann()
    model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model training
    model.fit(x_train_scaled, y_train, batch_size=42, epochs=100)

    # model prediction
    np.set_printoptions(precision=2)
    print(np.concatenate((model.predict(x_test_scaled).reshape(-1, 1), y_test.reshape(-1, 1)), 1))
    print(r2_score(y_test, model.predict(x_test_scaled)))


    