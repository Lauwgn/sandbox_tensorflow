import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.src import convert_vect_into_ids, correspondance_table, split_path_and_last_product

from models.mvisdense_manager import MvisDenseManager
from models.luw_manager import LuwManager


def main():
    luw = pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=100000)
    print(luw)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS ET AU MAX YYYYY PRODUITS                """
    """ ******************************************************************************** """

    visits_min_df = LuwManager.select_visitors_enough_visits(luw, 3, 10)
    print(visits_min_df)
    product_id_list = visits_min_df['product_id'].unique().tolist()
    nb_products_visits_min_df = len(product_id_list)

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE - UN ID_PRODUCT POUR UN INT                              """
    """ ******************************************************************************** """

    dict_products_corresp_int_id, dict_products_corresp_id_int = correspondance_table(product_id_list)

    with open("data/dict_products_corresp_int_id.pickle", 'wb') as file:
        pickle.dump(dict_products_corresp_int_id, file)

    with open("data/dict_products_corresp_id_int.pickle", 'wb') as file:
        pickle.dump(obj=dict_products_corresp_id_int, file=file)

    """ ******************************************************************************** """
    """ SPLIT : DATA // EXPECTED RESULT                                                  """
    """ ******************************************************************************** """

    visitors, luw_path_for_input, last_product_list = split_path_and_last_product(visits_min_df)

    print("Check equality {}, {}, {}".format(len(visitors), luw_path_for_input.index.nunique(), len(last_product_list)), '\n')

    """ ******************************************************************************** """
    """ PREPARATION DES DONNEES                                                          """
    """ ******************************************************************************** """

    luw_path_for_input.reset_index(inplace=True)
    mvis_input = MvisDenseManager.make_mvisdense(luw_path_for_input, product_id_list)
    mvis_input.rename_columns_to_int(dict_products_corresp_id_int)
    mvis_input = mvis_input.loc[visitors]
    # print(mvis_input)
    # mvis_input.to_csv("data/mvis_input.csv")


    """ ******************************************************************************** """
    """ ENTRAINEMENT DU MODELE                                                           """
    """ ******************************************************************************** """

    expected_list_int = list(map(lambda x: dict_products_corresp_id_int[x], last_product_list))
    print("Maximum des id dans last product seen : {}".format(np.max(expected_list_int)))
    print("Longueur de last product seen : {}".format(len(expected_list_int)))
    # print(last_product_list)
    # print(expected_list_int)
    X = mvis_input.values
    y = np.array(expected_list_int)
    # print(X, y, sep='\n')
    # print(len(np.unique(y)))

    X_train, X_test, y_train, y_test, vis_train, vis_test = train_test_split(X, y, visitors, test_size=0.2, random_state=10)
    # X_train, y_train, vis_train, = X, y, visitors
    print("X.shape : {}".format(X.shape),
          "X_train.shape : {}".format(X_train.shape),
          # "X_test.shape : {}".format(X_test.shape),
          "y.shape : {}".format(y.shape),
          "y_train.shape : {}".format(y_train.shape),
          # "y_test.shape : {}".format(y_test.shape),
          "visitors.shape : {}".format(visitors.shape),
          "vis_train.shape : {}".format(vis_train.shape),
          # "vis_test.shape : {}".format(vis_test.shape),
          '\n', sep='\n')
    # print(X_train)
    # print(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(nb_products_visits_min_df)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=True)
    # model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=True)

    model.save('data/model_azimut_v2')

    exit()

    """ ******************************************************************************** """
    """ RECHERCHE DES PARAMETRES nb_epoch ET batch_size                                  """
    """ ******************************************************************************** """

    loss_list, accuracy_list = [], []

    nb_epoch_min, nb_epoch_max, nb_epoch_step = 3, 20, 2
    # batch_size_min, batch_size_max, batch_size_step = 16, 128, 16

    # for batch_size in np.arange(batch_size_min, batch_size_max, batch_size_step):
    for nb_epoch in np.arange(nb_epoch_min, nb_epoch_max, nb_epoch_step):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(nb_products_visits_min_df)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # model.fit(X_train, y_train, batch_size=batch_size, epochs=9, verbose=False, use_multiprocessing=False)
        model.fit(X_train, y_train, batch_size=32, epochs=nb_epoch, verbose=False, use_multiprocessing=False)

        a = model.evaluate(X_test, y_test, verbose=False)
        loss_list.append(a[0])
        accuracy_list.append(a[1])
        print(nb_epoch, a)
        # print(batch_size, a)

    plt.plot(np.arange(nb_epoch_min, nb_epoch_max, nb_epoch_step), loss_list, c='b')
    plt.plot(np.arange(nb_epoch_min, nb_epoch_max, nb_epoch_step), accuracy_list, c='r')
    # plt.plot(np.arange(batch_size_min, batch_size_max, batch_size_step), loss_list, c='b')
    # plt.plot(np.arange(batch_size_min, batch_size_max, batch_size_step), accuracy_list, c='r')
    plt.show()

    """ ******************************************************************************** """
    """ CONTROLES DU MODELE                                                              """
    """ ******************************************************************************** """

    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #
    # predictions = probability_model.predict(X_test)

    # print('\n')
    # print(X_test[2])
    # print(predictions[2])
    # print(np.sum(predictions[2]))
    # print(np.argmax(predictions[2]))

    # for i in [2, 3, 4]:
    #     r = convert_vect_into_ids(X_test[i], dict_products_corresp_int_id)
    #     print("Visitor id : {}".format(vis_test[i]))
    #     print("Parcours : {}".format(r))
    #     print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
    #     print("Reality : {}".format(dict_products_corresp_int_id[y_test[i]]), '\n')


main()

