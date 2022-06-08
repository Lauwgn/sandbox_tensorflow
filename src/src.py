import numpy as np
import pandas as pd

from models.catalog_manager import CatalogManager

def correspondance_table(product_id_list):

    product_ids_df = pd.DataFrame(data=product_id_list, columns=["product_id"])
    product_ids_df = product_ids_df.drop_duplicates().reset_index(drop=True)
    # print(product_ids_df)
    # print(product_ids_df.index)

    dict_products_corresp_int_id = dict(zip(product_ids_df.index, product_ids_df['product_id']))
    dict_products_corresp_id_int = dict(zip(product_ids_df['product_id'], product_ids_df.index))
    # print(dict_products_corresp_int_id)
    # print(dict_products_corresp_id_int)

    return dict_products_corresp_int_id, dict_products_corresp_id_int


def convert_vect_into_ids(x, dict_products_corresp_int_id):
    a = np.where(x == 1)[0]
    # print(a)
    result = [dict_products_corresp_int_id[i] for i in a]
    return result


def convert_id_into_category(product_id, catalog):

    prod = CatalogManager.find_product_in_catalog_with_attributs(catalog, attribut="id", attr_value=product_id)
    # print(prod.to_dict())
    category = prod.convert_into_category_azimut()
    return category


def split_path_and_last_product(luw, is_test=False):
    """
    :param luw:
    :param is_test:
    :return:
        visitors : visitor's id list uniques, format numpy.array
        luw_input :  X, data for model
        expected_list : Y, data to predict
    """

    visitors, input_list, input_index, input_df_list, input_df_index, expected_list = [], [], [], [], [], []
    luw = luw.set_index(keys=['visitor_id'])[['product_id']]
    # print(visits_min_df)

    # for tmp_visitor in visits_min_df.index.unique()[:2]:
    for tmp_visitor in luw.index.unique():
        visitors.append(tmp_visitor)
        tmp_luw = luw.loc[tmp_visitor]
        # print(tmp_luw)

        input_index.append(tmp_visitor)
        input_list.append(tmp_luw['product_id'][:-1].to_list())

        input_df_index += [tmp_visitor for i in range(len(tmp_luw) - 1)]
        input_df_list += tmp_luw['product_id'][:-1].to_list()
        expected_list.append(tmp_luw['product_id'][-1])

    luw_input = pd.DataFrame(data=input_df_list, index=pd.Index(input_df_index, name='visitor_id'),
                             columns=['product_id'])
    # print('\n', luw_input)
    if not is_test:
        print("Nb de lignes dans luw_input + nb expected : {}".format(len(luw_input) + len(expected_list)),
              "Nb de visites luw min visites : {}".format(len(luw)),
              "Nb de produits dans luw min visites : {}".format(luw['product_id'].nunique()),
              '\n', sep='\n')

    return np.array(visitors), luw_input, expected_list



