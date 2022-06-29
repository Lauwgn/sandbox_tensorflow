import numpy as np
import pandas as pd

from models.catalog_manager import CatalogManager


def correspondance_table_product_id(product_id_list):

    product_ids_df = pd.DataFrame(data=product_id_list, columns=["product_id"])
    product_ids_df = product_ids_df.drop_duplicates().reset_index(drop=True)
    # print(product_ids_df)
    # print(product_ids_df.index)

    dict_products_corresp_int_id = dict(zip(product_ids_df.index, product_ids_df['product_id']))
    dict_products_corresp_id_int = dict(zip(product_ids_df['product_id'], product_ids_df.index))
    # print(dict_products_corresp_int_id)
    # print(dict_products_corresp_id_int)

    return dict_products_corresp_int_id, dict_products_corresp_id_int


def correspondance_table_category(category_id_list):

    product_cat_df = pd.DataFrame(data=category_id_list, columns=["category"])
    product_cat_df = product_cat_df.drop_duplicates().reset_index(drop=True)
    # print(product_cat_df)
    # print(product_cat_df.index)

    dict_products_corresp_int_cat = dict(zip(product_cat_df.index, product_cat_df['category']))
    dict_products_corresp_cat_int = dict(zip(product_cat_df['category'], product_cat_df.index))
    # print(dict_products_corresp_int_cat)
    # print(dict_products_corresp_cat_int)

    return dict_products_corresp_int_cat, dict_products_corresp_cat_int


def convert_vect_into_ids(x, dict_products_corresp_int_id):
    a = np.where(x == 1)[0]
    # print(a)
    result = [dict_products_corresp_int_id[i] for i in a]
    return result


def convert_id_into_category(product_id, catalog):

    category = None
    prod = CatalogManager.find_product_in_catalog_with_attributs(catalog, attribut="id", attr_value=product_id)
    # print(prod.to_dict())
    if prod:
        category = prod.convert_into_category_azimut()
    return category


def control_data(luw, X_train, X_test, y_train, y_test, vis_train, vis_test, dict_products_corresp_int_id):

    control_test_set = np.random.randint(0, len(vis_test), 3)
    control_test_ok = True

    if len(X_test) != len(y_test) or len(X_test) != len(vis_test):
        control_test_ok = False

    for i in control_test_set:
        tmp_vis = vis_test[i]
        tmp_luw = luw.loc[tmp_vis]
        luw_id_list = tmp_luw['product_id'].tolist()
        data_int_list = np.where(X_test[i] == 1)[0]
        data_id_list = [dict_products_corresp_int_id[tmp] for tmp in data_int_list]
        true_id = dict_products_corresp_int_id[y_test[i]]
        # print(set(data_id_list) == set(luw_id_list[:-1]))
        # print(true_id == luw_id_list[-1])
        if set(data_id_list) != set(luw_id_list[:-1]):
            control_test_ok = False
        if true_id != luw_id_list[-1]:
            control_test_ok = False

    # print(control_test_ok)

    control_train_set = np.random.randint(0, len(vis_train), 3)
    control_train_ok = True

    if len(X_train) != len(y_train) or len(X_train) != len(vis_train):
        control_train_ok = False

    for i in control_train_set:
        tmp_vis = vis_train[i]
        tmp_luw = luw.loc[tmp_vis]
        luw_id_list = tmp_luw['product_id'].tolist()
        data_int_list = np.where(X_train[i] == 1)[0]
        data_id_list = [dict_products_corresp_int_id[tmp] for tmp in data_int_list]
        true_id = dict_products_corresp_int_id[y_train[i]]
        # print(set(data_id_list) == set(luw_id_list[:-1]))
        # print(true_id == luw_id_list[-1])
        if set(data_id_list) != set(luw_id_list[:-1]):
            control_train_ok = False
        if true_id != luw_id_list[-1]:
            control_train_ok = False

    if control_train_ok and control_test_ok:
        print("Control data over train data and test data : ok")
    else:
        print('\n', "WARNING !!! Pb with data")


def search_max_occurences(tmp_list):

    df = pd.Series(tmp_list)
    occ_df = df.value_counts(sort=True, ascending=False)

    return occ_df.index[0]


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



