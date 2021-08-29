import numpy as np
import pandas as pd
import data_preprocessing as dp
import ann
import tensorflow as tf

if __name__ == '__main__':

    # data preprocessing
    data = pd.read_excel("data/Folds5x2_pp.xlsx")
    data_preprocessor = dp.DataPreprocessing(data)
    data_preprocessor.split_data(test_size=0.2)
    x_train, x_test, y_train, y_test = data_preprocessor.get_processed_data()
    x_train_scaled, x_test_scaled = data_preprocessor.rescaled_data()

    # model creation
    model = ann.create_ann()
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics="accuracy")

    # model training



    