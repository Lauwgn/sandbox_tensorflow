import numpy as np
import pandas as pd
import sklearn.metrics
import pickle
from matplotlib import pyplot as plt

from src.src import control_data, correspondance_table_product_id, correspondance_table_category, split_path_and_last_product
from src.train_to_predict_id import train_to_predict_id
from src.train_to_predict_category import train_to_predict_category

from models.models import Luw
from models.catalog_manager import CatalogManager
from models.mvisdense_manager import MvisDenseManager
from models.luw_manager import LuwManager


def main():

    luw = Luw(pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=100000))
    # print(luw)

    catalog = CatalogManager.import_from_json("data/20211206-catalog-533d1d6652e1-fr-en.json")
    luw.filter_product_ids_from_catalog(catalog)
    # print(luw)

    catalog_df = pd.read_csv("data/catalog_azimut_cat_revu.csv")
    catalog_df.set_index(keys="product_id", inplace=True)
    # print(catalog_df)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS ET AU MAX YYYYY PRODUITS                """
    """ ******************************************************************************** """

    visits_min_df = LuwManager.select_visitors_enough_visits(luw, 3, 10)
    # print(visits_min_df)
    product_id_list = visits_min_df['product_id'].unique().tolist()
    nb_products_visits_min_df = len(product_id_list)

    """ ******************************************************************************** """
    """ SPLIT : DATA // EXPECTED RESULT                                                  """
    """ ******************************************************************************** """

    visitors, luw_path_for_input, last_product_list = split_path_and_last_product(visits_min_df)

    print("Check equality {}, {}, {}".format(len(visitors), luw_path_for_input.index.nunique(), len(last_product_list)), '\n')

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE - UN ID_PRODUCT POUR UN INT                              """
    """ ******************************************************************************** """

    dict_products_corresp_int_id, dict_products_corresp_id_int = correspondance_table_product_id(product_id_list)

    with open("data/models_category/dict_products_corresp_int_id.pickle", 'wb') as file:
        pickle.dump(dict_products_corresp_int_id, file)

    with open("data/models_category/dict_products_corresp_id_int.pickle", 'wb') as file:
        pickle.dump(obj=dict_products_corresp_id_int, file=file)

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE - UNE CATEGORIE POUR UN INT                              """
    """ ******************************************************************************** """

    nb_category = catalog_df["category"].nunique()
    expected_list_category = list(map(lambda x: catalog_df["category"].loc[x], last_product_list))

    dict_products_corresp_int_cat, dict_products_corresp_cat_int = correspondance_table_category(expected_list_category)

    with open("data/models_category/dict_products_corresp_int_cat.pickle", 'wb') as file:
        pickle.dump(dict_products_corresp_int_cat, file)

    with open("data/models_category/dict_products_corresp_cat_int.pickle", 'wb') as file:
        pickle.dump(obj=dict_products_corresp_cat_int, file=file)

    expected_list_category_int = list(map(lambda x: dict_products_corresp_cat_int[x], expected_list_category))
    # print(expected_list_category_int)

    """ ******************************************************************************** """
    """ PREPARATION DES DONNEES                                                          """
    """ ******************************************************************************** """

    luw_path_for_input.reset_index(inplace=True)
    mvis_input = MvisDenseManager.make_mvisdense(luw_path_for_input, product_id_list)
    mvis_input.rename_columns_to_int(dict_products_corresp_id_int)
    mvis_input = mvis_input.loc[visitors]
    # print(mvis_input)
    # mvis_input.to_csv("data/mvis_input.csv")

    train_to_predict_category(mvis_input, visitors, expected_list_category_int, nb_category)


main()

