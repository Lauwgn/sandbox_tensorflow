import numpy as np
import pandas as pd


def correspondance_table(product_id_list):

    product_ids_df = pd.DataFrame(data=product_id_list, columns=["product_id"])
    product_ids_df = product_ids_df.drop_duplicates().reset_index(drop=True)
    # print(product_ids_df)
    print(product_ids_df.index)

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


def make_mvis(luw, product_id_list):
    luw['nb_visit'] = np.ones(shape=(len(luw), 1))
    mvis = pd.pivot_table(luw, index=["visitor_id"], columns=['product_id'], values='nb_visit',
                          fill_value=0.0)
    # print(mvis_3_input)

    products_to_add = []
    for tmp_id in product_id_list:
        if tmp_id not in mvis.columns:
            products_to_add.append(tmp_id)

    for tmp_id in products_to_add:
        mvis[tmp_id] = np.zeros(shape=(len(mvis), 1))


    if mvis.sum().sum() != (len(luw)):
          print("ERROR : loss of data - see code for more information")

    if len(np.intersect1d(mvis.columns, product_id_list)) != len(product_id_list):
          print("ERROR : loss of data - see code for more information")

    # print(mvis)
    # print(mvis.columns)

    return mvis


def mvis_rename_columns(mvis, dict_products_corresp_id_int):

    new_index = mvis.columns.copy()
    new_index = new_index.map(lambda x: dict_products_corresp_id_int[x])
    mvis.columns = new_index
    # print(new_index)

    mvis.sort_index(axis='columns', ascending=True, inplace=True)

    # print(mvis)
    # print(mvis.columns)

    return mvis.copy()

def select_visitors_enough_visits(luw, min_visits, max_visits):

    nb_visits_series = luw['visitor_id'].value_counts().sort_values(ascending=False)
    nb_visits_series = nb_visits_series[nb_visits_series <= max_visits]
    # print(nb_visits_series)

    visitor_id_min_visits_list = nb_visits_series[nb_visits_series >= min_visits].index.tolist()
    # print(visitor_id_min_visits_list)
    print('\n',
          "Nb visitors with more than {} visits and less than {} visits : {}".format(min_visits, max_visits,
                                                                                     len(visitor_id_min_visits_list)),
          "Nb products in Luw : {}".format(luw['product_id'].nunique()), '\n', sep='\n')

    visits_min_df = luw.set_index(keys=['visitor_id']).loc[visitor_id_min_visits_list][['product_id']].copy()
    # print('\n', visits_min_df)

    return visits_min_df


def split_path_and_last_product(visits_min_df):

    visitors, input_list, input_index, input_df_list, input_df_index, expected_list = [], [], [], [], [], []

    # for tmp_visitor in visits_min_df.index.unique()[:2]:
    for tmp_visitor in visits_min_df.index.unique():
        visitors.append(tmp_visitor)
        tmp_luw = visits_min_df.loc[tmp_visitor]
        # print(tmp_luw)

        input_index.append(tmp_visitor)
        input_list.append(tmp_luw['product_id'][:-1].to_list())

        input_df_index += [tmp_visitor for i in range(len(tmp_luw) - 1)]
        input_df_list += tmp_luw['product_id'][:-1].to_list()
        expected_list.append(tmp_luw['product_id'][-1])

    visits_min_df_input = pd.DataFrame(data=input_df_list, index=pd.Index(input_df_index, name='visitor_id'),
                                       columns=['product_id'])
    # print('\n', visits_3min_df_input)
    print("Nb de lignes dans luw_input + nb expected : {}".format(len(visits_min_df_input) + len(expected_list)),
          "Nb de visites luw min visites : {}".format(len(visits_min_df)),
          "Nb de produits dans luw min visites : {}".format(visits_min_df['product_id'].nunique()),
          '\n', sep='\n')

    return np.array(visitors), visits_min_df_input, expected_list



