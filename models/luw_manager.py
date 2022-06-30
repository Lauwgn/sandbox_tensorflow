from scipy import sparse
import numpy as np
import pandas as pd
from models.models import Luw, Mvis


class LuwManager:

    @staticmethod
    def import_from_pandas(filename):
        #Luw from pipelines
        # Two columns, one visitor_id, the other one product_id
        luw = pd.read_csv(filename)
        luw.rename(columns={"wvi": "visitor_id", "id": "product_id"},
                   inplace=True, errors="ignore")
        luw.dropna(how='any', inplace=True)
        luw.drop_duplicates(inplace=True)
        return Luw(luw)

    @staticmethod
    def select_products_and_visitors_frequentation(l_uw, mvis_thresholds):

        # Select only product_ids with many visits
        nb_visits_min_threshold_id = mvis_thresholds.nb_visits_min_threshold_id
        nb_visits_max_threshold_id = mvis_thresholds.nb_visits_max_threshold_id

        l_uw = LuwManager.select_product_ids_enough_visits(l_uw, nb_visits_min_threshold_id, nb_visits_max_threshold_id)

        # Select only visitors with many visits
        nb_visits_min_threshold_visitors = mvis_thresholds.nb_visits_min_threshold_visitors
        nb_visits_max_threshold_visitors = mvis_thresholds.nb_visits_max_threshold_visitors

        l_uw = LuwManager.select_visitors_enough_visits(l_uw, nb_visits_min_threshold_visitors, nb_visits_max_threshold_visitors)

        return l_uw

    @staticmethod
    def select_product_ids_enough_visits(luw, nb_visits_min_threshold_id, nb_visits_max_threshold_id,
                                         is_test=False):

        if nb_visits_max_threshold_id is None:
            nb_visits_max_threshold_id = len(luw)

        nb_visits_df = luw.groupby(by=['product_id']).size().reset_index()
        nb_visits_df.set_index(keys='product_id', inplace=True, drop=True)
        nb_visits_df.rename(columns={0: 'nb_visits'}, inplace=True)

        id_list = nb_visits_df[(nb_visits_df['nb_visits'] >= nb_visits_min_threshold_id)
                               & (nb_visits_df['nb_visits'] <= nb_visits_max_threshold_id)].index

        luw.set_index(keys='product_id', inplace=True, drop=True)
        new_luw = luw.copy()
        new_luw = new_luw.loc[id_list]
        new_luw.reset_index(inplace=True)

        return new_luw

    @staticmethod
    def select_visitors_enough_visits(luw, nb_visits_min_threshold_visitors, nb_visits_max_threshold_visitors, is_test=False):

        if nb_visits_max_threshold_visitors is None:
            nb_visits_max_threshold_visitors = len(luw)

        nb_visits_df = luw.groupby(by=['visitor_id']).size().reset_index()
        nb_visits_df.set_index(keys='visitor_id', inplace=True, drop=True)
        nb_visits_df.rename(columns={0: 'nb_visits'}, inplace=True)

        visitor_id_min_visits_list = nb_visits_df[(nb_visits_df['nb_visits'] >= nb_visits_min_threshold_visitors)
                                & (nb_visits_df['nb_visits'] <= nb_visits_max_threshold_visitors)].index

        luw.set_index(keys='visitor_id', inplace=True, drop=True)
        new_luw = luw.copy()
        new_luw = new_luw.loc[visitor_id_min_visits_list]
        new_luw.reset_index(inplace=True)
        if not is_test:
            print('\n',
                  "Nb visitors with more than {} visits and less than {} visits : {}".format(
                      nb_visits_min_threshold_visitors, nb_visits_max_threshold_visitors,
                      len(visitor_id_min_visits_list)),
                  "Nb products in Luw : {}".format(luw['product_id'].nunique()), '\n', sep='\n')

        return Luw(new_luw)

    @staticmethod
    def convert_luw_into_mvis_sparse(l_uw):

        """
        input : l_uw : pandas.DataFrame, 2 columns : 'id' and 'wvi' - <type : wag_recommandation.l_uw>

        :return: m_vis : <type : wag_recommandation.m_vis>
        { wvi : list of wvi sorted according to the sparse matrix indices ;
          id  : list of ids according to the sparse matrix indices ;
          values : scipy.sparse.matrix.csrmatrix()
                a row -> a wvi
                a column -> an id
                shape as sparse matrix, with only 1.
            - if wvi xxxx has visited id yyyyy then the value is set to 1
        }
        """

        if len(l_uw) == 0:
            m_vis = Mvis(visitor_ids_list=[],
                         product_ids_list=[],
                         values=None)

        else:
            list_wvi = l_uw['visitor_id'].sort_values().unique()
            list_ids = l_uw['product_id'].unique()

            l_uw.sort_values(by='visitor_id', inplace=True)

            l_uw['row_id_bool'] = l_uw['visitor_id'] == l_uw['visitor_id'].shift(1).fillna(l_uw['visitor_id'])

            l_uw.iloc[0, l_uw.columns.get_loc(
                'row_id_bool')] = False  # Sinon on perd le premier moment du l_uw, avec le premier wvi
            l_uw['row_id'] = (~l_uw['row_id_bool']).cumsum() - 1
            # print(l_uw[['wvi', 'row_id_bool', 'row_id']].head(50))

            l_uw['col_id'] = l_uw['product_id'].apply(lambda x: list(list_ids).index(x))
            # print(l_uw[['id', 'col_id']].head(50))

            values = np.array([1] * l_uw.shape[0])

            # print("Longueur de produits equivalent col : {}".format(len(list_ids)))
            # print("Longueur de visiteur équivalent row : {}".format(len(list_wvi)))

            sp_mvis_f = sparse.coo_matrix((values, (l_uw['row_id'].tolist(), l_uw['col_id'].tolist())),
                                          shape=(len(list_wvi), len(list_ids)))

            m_vis = Mvis(visitor_ids_list=list_wvi.tolist(),
                         product_ids_list=list_ids.tolist(),
                         values=sp_mvis_f.tocsr())

        return m_vis

    @staticmethod
    def filter_list_with_only_ids_in_luw(luw, id_list):

        luw_id_list = luw["product_id"].unique()
        new_id_list = np.intersect1d(luw_id_list, id_list).tolist()

        return new_id_list
