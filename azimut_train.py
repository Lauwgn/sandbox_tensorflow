import numpy as np
import pandas as pd
import sklearn.metrics
import pickle
from matplotlib import pyplot as plt

from src.src import control_data, correspondance_table_product_id, split_path_and_last_product
from src.train_to_predict_id import train_to_predict_id
from src.train_to_predict_category import train_to_predict_category

from models.mvisdense_manager import MvisDenseManager
from models.luw_manager import LuwManager


def main():

    luw = pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=30000)
    # print(luw)

    catalog_df = pd.read_csv("data/catalog_azimut_cat.csv")
    catalog_df.set_index(keys="product_id", inplace=True)
    print(catalog_df)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS ET AU MAX YYYYY PRODUITS                """
    """ ******************************************************************************** """

    visits_min_df = LuwManager.select_visitors_enough_visits(luw, 3, 10)
    # print(visits_min_df)
    product_id_list = visits_min_df['product_id'].unique().tolist()
    nb_products_visits_min_df = len(product_id_list)

    """ ******************************************************************************** """
    """ TABLE DE CORRESPONDANCE - UN ID_PRODUCT POUR UN INT                              """
    """ ******************************************************************************** """

    dict_products_corresp_int_id, dict_products_corresp_id_int = correspondance_table_product_id(product_id_list)

    # with open("data/dict_products_corresp_int_id.pickle", 'wb') as file:
    #     pickle.dump(dict_products_corresp_int_id, file)
    #
    # with open("data/dict_products_corresp_id_int.pickle", 'wb') as file:
    #     pickle.dump(obj=dict_products_corresp_id_int, file=file)

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

    # train_to_predict_id(luw, mvis_input, visitors, last_product_list, nb_products_visits_min_df,
    #                     dict_products_corresp_int_id, dict_products_corresp_id_int)

    train_to_predict_category(luw, mvis_input, visitors, last_product_list, nb_products_visits_min_df,
                              catalog_df)

main()

