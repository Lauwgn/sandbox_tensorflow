import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

import tensorflow as tf
from tensorflow import keras

from models.models import Luw
from src.src import split_path_and_last_product, split_path_and_two_last_products
from src.determine_main_category import determine_main_category_luw_sorted
from src.control_by_cohort import control_by_cohort
from models.luw_manager import LuwManager
from models.mvisdense_manager import MvisDenseManager
from models.catalog_manager import CatalogManager

""" Le but est de comparer le modèle du réseau de neurone avec un modèle trivial"""


def main():

    """ ******************************************************************************** """
    """ TABLES DE CORRESPONDANCE                                                          """
    """ ******************************************************************************** """

    with open("data/models_category/dict_products_corresp_id_int.pickle", 'rb') as file:
        dict_products_corresp_id_int = pickle.load(file)

    with open("data/models_category/dict_products_corresp_int_id.pickle", 'rb') as file:
        dict_products_corresp_int_id = pickle.load(file)

    with open("data/models_category/dict_products_corresp_cat_int.pickle", 'rb') as file:
        dict_products_corresp_cat_int = pickle.load(file)

    with open("data/models_category/dict_products_corresp_int_cat.pickle", 'rb') as file:
        dict_products_corresp_int_cat = pickle.load(file)

    """ ******************************************************************************** """
    """ IMPORT LUW AND CATALOG                                                           """
    """ ******************************************************************************** """

    luw = Luw(pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=None))
    luw = Luw(luw[100000:].reset_index(drop=True))

    catalog = CatalogManager.import_from_json("data/20211206-catalog-533d1d6652e1-fr-en.json")
    luw.filter_product_ids_from_catalog(catalog)
    # print(luw)

    catalog_df = pd.read_csv("data/catalog_azimut_cat_revu.csv")
    catalog_df.set_index(keys="product_id", inplace=True)
    # print(catalog_df)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
    """ ******************************************************************************** """

    product_id_list = list(dict_products_corresp_int_id.values())
    # print(dict_products_corresp_id_int, dict_products_corresp_int_id)

    luw.filter_product_ids_in_list_of_ids(product_id_list)
    # print(luw)

    visits_min_df = LuwManager.select_visitors_enough_visits(luw, 3, 10)
    # print(visits_min_df)

    """ ******************************************************************************** """
    """ TRAVAIL SUR LE MODELE                                                          """
    """ ******************************************************************************** """

    model_basic(visits_min_df, catalog_df, dict_products_corresp_cat_int, dict_products_corresp_int_cat)

    # model_most_representated_category(visits_min_df, catalog_df, dict_products_corresp_cat_int, dict_products_corresp_int_cat)


def model_most_representated_category(visits_min_df, catalog_df, dict_products_corresp_cat_int, dict_products_corresp_int_cat):
    # Le modèle retient la catégorie la plus représentée dans le parcours de visite.
    # En cas d'égalité, c'est la dernière que l'on retient

    """ ******************************************************************************** """
    """ ADD COLUMN REF TO LUW                                                            """
    """ ******************************************************************************** """

    visits_min_df.add_column_category(catalog_df)
    visits_min_df = visits_min_df.set_index(keys="visitor_id")
    # print(visits_min_df)
    # print(catalog_df)

    """ ******************************************************************************** """
    """ SPLIT : DATA // EXPECTED RESULT                                                  """
    """ ******************************************************************************** """

    visitors, luw_path, last_product_list = split_path_and_last_product(visits_min_df)
    print("Check equality {}, {}, {}".format(len(visitors), luw_path.index.nunique(), len(last_product_list),
          len(last_product_list)), '\n')

    """ ******************************************************************************** """
    """ PREPROCESSING                                                                    """
    """ ******************************************************************************** """

    # luw_path.reset_index(inplace=True)
    # mvis_input = MvisDenseManager.make_mvisdense(luw_path, product_id_list)
    # mvis_input.rename_columns_to_int(dict_products_corresp_id_int)
    # mvis_input = mvis_input.loc[visitors]
    # x_test = mvis_input.values

    expected_list_category = list(map(lambda x: catalog_df["category"].loc[x], last_product_list))
    # print(expected_list_category)
    expected_list_category_int = list(map(lambda x: dict_products_corresp_cat_int[x], expected_list_category))
    y_true = np.array(expected_list_category_int)

    """ ******************************************************************************** """
    """ TRAVAIL SUR LE MODELE                                                           """
    """ ******************************************************************************** """

    dict_id_to_category = dict(zip(catalog_df.index, catalog_df['category']))
    # print(dict_id_to_category)

    y_pred = []
    for count, current_wvi in enumerate(tqdm(visits_min_df.index.unique())):
        current_wvi_path_df = visits_min_df.loc[[current_wvi]]
        # print(current_wvi_path_df['category'])
        current_category = determine_main_category_luw_sorted(current_wvi_path_df['product_id'].to_list(), dict_id_to_category)
        # print(current_category)
        y_pred.append(current_category)

    # print(y_pred)
    y_pred = list(map(lambda x: dict_products_corresp_cat_int[x], y_pred))

    print("Longueur de y_pred : {}, Longueur de y_true : {}".format(len(y_pred), len(y_true)))

    acc = tf.keras.metrics.Accuracy()
    print("Accuracy y_true, y_pred : {}".format(acc(y_pred, y_true).numpy()))

    cm = confusion_matrix(y_pred, y_true)
    print('\n', cm)
    print(dict_products_corresp_int_cat)

    """ ******************************************************************************** """
    """ PLOT : MATRICE DE CONFUSION                                                      """
    """ ******************************************************************************** """

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()

    classes = [dict_products_corresp_int_cat[i] for i in range(len(dict_products_corresp_int_cat))]

    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.show()


def model_basic(visits_min_df, catalog_df, dict_products_corresp_cat_int, dict_products_corresp_int_cat):

    """ ******************************************************************************** """
    """ SPLIT : DATA // EXPECTED RESULT                                                  """
    """ ******************************************************************************** """

    visitors, luw_path, prev_last_product_list, last_product_list = split_path_and_two_last_products(visits_min_df)
    print("Check equality {}, {}, {}, {}".format(len(visitors), luw_path.index.nunique(), len(last_product_list),
          len(prev_last_product_list)), '\n')

    """ ******************************************************************************** """
    """ PREPROCESSING                                                                    """
    """ ******************************************************************************** """

    # luw_path.reset_index(inplace=True)
    # mvis_input = MvisDenseManager.make_mvisdense(luw_path, product_id_list)
    # mvis_input.rename_columns_to_int(dict_products_corresp_id_int)
    # mvis_input = mvis_input.loc[visitors]
    # x_test = mvis_input.values

    expected_list_category = list(map(lambda x: catalog_df["category"].loc[x], last_product_list))
    expected_list_category_int = list(map(lambda x: dict_products_corresp_cat_int[x], expected_list_category))
    y_true = np.array(expected_list_category_int)

    """ ******************************************************************************** """
    """ TRAVAIL SUR LE MODELE 1                                                           """
    """ ******************************************************************************** """
    # Le modèle retient la catégorie de l'avant-dernier produit vu.
    # Le modèle prévoit que la catégorie du dernier produit et celle de l'avant-dernier produit sont les mêmes

    prev_last_product_category_list = list(map(lambda x: catalog_df["category"].loc[x], prev_last_product_list))
    prev_last_product_category_list_int = list(map(lambda x: dict_products_corresp_cat_int[x], prev_last_product_category_list))
    y_pred = np.array(prev_last_product_category_list_int)
    print("Longueur de y_pred : {}, Longueur de y_true : {}".format(len(y_pred), len(y_true)))

    acc = tf.keras.metrics.Accuracy()
    print("Accuracy y_true, y_pred : {}".format(acc(y_true, y_pred).numpy()))

    cm = confusion_matrix(y_pred, y_true)
    print('\n', cm)
    print(dict_products_corresp_int_cat)

    """ ******************************************************************************** """
    """ PLOT : MATRICE DE CONFUSION                                                      """
    """ ******************************************************************************** """

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()

    classes = [dict_products_corresp_int_cat[i] for i in range(len(dict_products_corresp_int_cat))]

    # tick_marks, classes = np.arange(cm.shape[0]), np.arange(cm.shape[0])
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.show()


main()
