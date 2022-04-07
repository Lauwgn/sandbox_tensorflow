import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.src import convert_vect_into_ids, correspondance_table, select_visitors_enough_visits, split_data_and_expected_result


def main():

    luw = pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=5000)
    product_id_list = luw['product_id'].unique().tolist()
    nb_products_luw = len(product_id_list)
    # print(luw)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
    """ ******************************************************************************** """

    visits_min_df = select_visitors_enough_visits(luw, 3, 10)

    """ ******************************************************************************** """
    """ SPLIT : DATA // EXPECTED RESULT                                                  """
    """ ******************************************************************************** """

    visitors, visits_min_df_input, expected_list = split_data_and_expected_result(visits_min_df)

    """ ******************************************************************************** """
    """ PREPARATION DES DONNEES                                                          """
    """ ******************************************************************************** """

    visits_min_df_input.reset_index(inplace=True)
    visits_min_df_input['nb_visit'] = np.ones(shape=(len(visits_min_df_input), 1))
    mvis_3_input = pd.pivot(visits_min_df_input, index=["visitor_id"], columns=['product_id'], values=['nb_visit'])
    # mvis_3_input = mvis_3_input.fillna(0).convert_dtypes()
    mvis_3_input = mvis_3_input.fillna(0)
    # print(mvis_3_input)

    if mvis_3_input.sum().sum() != (len(visits_min_df_input)):
          print("ERROR : loss of data - see code for more information")
    mvis_3_input.to_csv("data/mvis_3.csv")

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE                                                          """
    """ ******************************************************************************** """

    dict_products_corresp_int_id, dict_products_corresp_id_int = correspondance_table(product_id_list)

    """ ******************************************************************************** """
    """ ENTRAINEMENT DU MODELE                                                           """
    """ ******************************************************************************** """

    expected_list_int = list(map(lambda x: dict_products_corresp_id_int[x], expected_list))
    # print(expected_list)
    # print(expected_list_int)
    X = mvis_3_input.values
    y = np.array(expected_list_int)
    # print(X, y, sep='\n')
    # print(len(np.unique(y)))

    X_train, X_test, y_train, y_test, vis_train, vis_test = train_test_split(X, y, visitors, test_size=0.2, random_state=42)
    print("X_test.shape : {}".format(X_test.shape),
          "y_test.shape : {}".format(y_test.shape),
          "X_train.shape : {}".format(X_train.shape),
          "y_train.shape : {}".format(y_train.shape), '\n', sep='\n')
    # print(X_train)
    # print(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(len(np.unique(y)))
        tf.keras.layers.Dense(nb_products_luw)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, epochs=15)

    """ ******************************************************************************** """
    """ CONTROLES DU MODELE                                                           """
    """ ******************************************************************************** """

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(X_test)

    print('\n')
    # print(X_test[2])
    # print(predictions[2])
    # print(np.sum(predictions[2]))
    # print(np.argmax(predictions[2]))

    for i in [10, 15, 20]:
        r = convert_vect_into_ids(X_test[i], dict_products_corresp_int_id)
        print("Visitor id : {}".format(vis_test[i]))
        print("Parcours : {}".format(r))
        print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
        print("Reality : {}".format(dict_products_corresp_int_id[y_test[i]]), '\n')


main()

