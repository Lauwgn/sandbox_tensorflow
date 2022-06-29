import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.src import convert_vect_into_ids, correspondance_table_category
from sklearn.model_selection import train_test_split


def train_to_predict_category(luw, mvis_input, visitors, last_product_list, nb_products_visits_min_df,
                              catalog_df):

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE - UN ID_PRODUCT POUR UN INT                              """
    """ ******************************************************************************** """

    # print(last_product_list)
    expected_list_category = list(map(lambda x: catalog_df["category"].loc[x], last_product_list))
    dict_products_corresp_int_cat, dict_products_corresp_cat_int = correspondance_table_category(expected_list_category)
    expected_list_category_int = list(map(lambda x: dict_products_corresp_cat_int[x], expected_list_category))
    # print(expected_list_category_int)

    """ ******************************************************************************** """
    """ PREPROCESSING                                                                    """
    """ ******************************************************************************** """
    nb_category = catalog_df["category"].nunique()

    x = mvis_input.values
    y = np.array(expected_list_category_int)

    x_train, x_test, y_train, y_test, vis_train, vis_test = train_test_split(x, y, visitors, test_size=0.2, random_state=20)
    # x_train, y_train, vis_train, = X, y, visitors
    print("x.shape : {}".format(x.shape),
          "x_train.shape : {}".format(x_train.shape),
          # "x_test.shape : {}".format(x_test.shape),
          "y.shape : {}".format(y.shape),
          "y_train.shape : {}".format(y_train.shape),
          # "y_test.shape : {}".format(y_test.shape),
          "visitors.shape : {}".format(visitors.shape),
          "vis_train.shape : {}".format(vis_train.shape),
          # "vis_test.shape : {}".format(vis_test.shape),
          '\n', sep='\n')
    # print(x_train)
    # print(y_train)

    # control_data(luw, x_train, x_test, y_train, y_test, vis_train, vis_test, dict_products_corresp_int_id)

    """ ******************************************************************************** """
    """ ENTRAINEMENT DU MODELE                                                           """
    """ ******************************************************************************** """

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(nb_category)
    ])

    q = model.compile(
                  optimizer='adam',
                  # optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )

    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100, verbose=False)
    # model.fit(x_train, y_train, batch_size=128, epochs=40, verbose=True)


    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(r.history['accuracy'], label='accuracy')
    plt.plot(r.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    exit()


    # @todo : ON EN EST LA !!!!!!!!!!!


    # model.save('data/model_azimut_v2')

    """ ******************************************************************************** """
    """ CONTROLES DU MODELE                                                              """
    """ ******************************************************************************** """

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(x_test)

    print('\n')
    print(x_test[2])
    print(predictions[2])
    print(np.sum(predictions[2]))
    print(np.argmax(predictions[2]))

    for i in [2, 3, 4]:
        r = convert_vect_into_ids(x_test[i], dict_products_corresp_int_id)
        print("Visitor id : {}".format(vis_test[i]))
        print("Parcours : {}".format(r))
        print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
        print("Reality : {}".format(dict_products_corresp_int_id[y_test[i]]), '\n')


    # exit()

    """ ******************************************************************************** """
    """ RECHERCHE DES PARAMETRES nb_epoch ET batch_size                                  """
    """ ******************************************************************************** """

    loss_list, accuracy_list = [], []

    nb_epoch_min, nb_epoch_max, nb_epoch_step = 3, 50, 3
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

        # model.fit(x_train, y_train, batch_size=batch_size, epochs=9, verbose=False, use_multiprocessing=False)
        model.fit(x_train, y_train, batch_size=32, epochs=nb_epoch, verbose=False, use_multiprocessing=False)

        a = model.evaluate(x_test, y_test, verbose=False)
        loss_list.append(a[0])
        accuracy_list.append(a[1])
        # print(nb_epoch, a)
        # print(batch_size, a)

    plt.plot(np.arange(nb_epoch_min, nb_epoch_max, nb_epoch_step), loss_list, c='b', label='loss')
    plt.plot(np.arange(nb_epoch_min, nb_epoch_max, nb_epoch_step), 10 * np.array(accuracy_list), c='r', label='10 x accuracy')
    # plt.plot(np.arange(batch_size_min, batch_size_max, batch_size_step), loss_list, c='b')
    # plt.plot(np.arange(batch_size_min, batch_size_max, batch_size_step), accuracy_list, c='r')
    plt.legend()
    plt.show()