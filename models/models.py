from datetime import datetime
import pandas as pd
import numpy as np
import json

from wag_core_modules.ia.cohort_matrix import CohortMatrix
from tqdm import tqdm
tqdm.pandas()


class Curlr(pd.DataFrame):
    #todo : verification import/export
    dict_types = {'product_id': str, 'cohort_id': str, 'qualif': str, 'url': str, 'name': str, 'name_cleaned': object,
                  'name_cleaned_freq': object, 'template': str, 'template_freq': object, 'keywords': object,
                  'keywords_freq': object, 'price': float, 'nb_visitors_id': int, 'nb_visitors_coh': int,
                  'cohort_size': int, 'ranking': int}
    ordering_column = ['cohort_id', 'qualif', 'url', 'name', 'name_cleaned', 'name_cleaned_freq', 'template',
                       'template_freq', 'keywords', 'keywords_freq', 'price', 'nb_visitors_id', 'nb_visitors_coh',
                       'cohort_size', 'ranking']

    def change_types(self):
        tmp_curlr = self.copy()
        tmp_curlr.reset_index(inplace=True)

        for curr_column, curr_type in Curlr.dict_types.items():
            if curr_column in tmp_curlr.columns:
                if not isinstance(tmp_curlr[curr_column].dtype, curr_type):
                    tmp_curlr = tmp_curlr.astype(dtype={curr_column: curr_type}, copy=True)

        if "product_id" in tmp_curlr.columns:
            tmp_curlr.set_index(keys='product_id', inplace=True)
        elif "id" in tmp_curlr.columns:
            tmp_curlr.set_index(keys='id', inplace=True)
        else:
            raise ValueError("No columns id or product_id in the dataframe")
        return Curlr(tmp_curlr)

    def ordering_columns(self):
        tmp_self = self.reindex(columns=Curlr.ordering_column)
        return tmp_self

    def check_types(self):

        for curr_column, curr_type in Curlr.dict_types.items():
            if curr_column in self.columns:
                if not isinstance(self[curr_column].dtype, curr_type):
                    return False
        return True

    def controls_curlr_info(self):
        tmp_c_url_r = self.copy()
        nb_vis_id = np.sum(tmp_c_url_r['nb_visitors_id'])

        tmp_c_url_r.reset_index(inplace=True)
        tmp_c_url_r.set_index(keys='cohort_id', inplace=True, drop=True)

        tmp_list = []

        for tmp_coh in tmp_c_url_r.index.unique():
            if isinstance(tmp_c_url_r['nb_visitors_coh'].loc[tmp_coh], np.int64):
                tmp_value = tmp_c_url_r['nb_visitors_coh'].loc[tmp_coh]
            elif isinstance(tmp_c_url_r['nb_visitors_coh'].loc[tmp_coh], np.int32):
                tmp_value = tmp_c_url_r['nb_visitors_coh'].loc[tmp_coh]
            else:
                tmp_value = tmp_c_url_r['nb_visitors_coh'].loc[tmp_coh].values[0]
            tmp_list.append(tmp_value)

        nb_vis_coh = np.sum(tmp_list)

        return nb_vis_coh, nb_vis_id

    def display_end(self):
        if not self.empty:  # On vérifie que l'objet curlr n'est pas vide
            nb_vis_coh, nb_vis_url = self.controls_curlr_info()
            print('\n', "Controls :")
            print('Nb total of visitors in cohorts : %s' % nb_vis_coh)
            print('Nb total of visits on all url : %s' % nb_vis_url)

    def display_import_curlr(self):
        print('\n', "Number of id : " + str(len(self.index)))
        print('\n', "Number of cohorts : " + str(len(self['cohort_id'].unique())))
        print('\n', "Import Curl,r - done", '\n')

    def update_nb_visitors_id_from_luw(self, l_uw):

        product_groups = l_uw.groupby(by=["product_id"]).size()
        # print(product_groups)
        current_product_ids_list = np.intersect1d(product_groups.index, self.index)
        self.loc[current_product_ids_list, 'nb_visitors_id'] = product_groups.loc[current_product_ids_list]

    def update_cohort_size(self):
        if 'cohort_id' in self.columns:
            groups = self.groupby(by='cohort_id').size()
            cohort_size_list = [groups.loc[tmp_coh] for tmp_coh in self['cohort_id']]
            self["cohort_size"] = cohort_size_list

    def sort_for_export_cohort_size_and_nb_visitors_id(self):
        self.sort_values(by=["cohort_size", "cohort_id", "nb_visitors_id"], ascending=[False, True, False], inplace=True)

    def sort_for_export_full(self):

        if "nb_visitors_coh" in self.columns and "nb_visitors_id" in self.columns:
            self.sort_values(by=["nb_visitors_coh", "cohort_size", "cohort_id", "nb_visitors_id"],
                             ascending=[False, False, True, False], inplace=True)
        else:
            if "nb_visitors_coh" in self.columns:
                self.sort_values(by=["nb_visitors_coh", "cohort_size", "cohort_id"],
                                 ascending=[False, False, True], inplace=True)
            elif "nb_visitors_id" in self.columns:
                self.sort_values(by=["cohort_size", "cohort_id", "nb_visitors_id"],
                                 ascending=[False, True, False], inplace=True)
            else:
                self.sort_values(by=["cohort_size", "cohort_id"],
                                 ascending=[False, True], inplace=True)

    def export_c_url_r(self, curlr_export_filename, current_wti):
        """Export from a curlr Object

        Parameters
        ----------
        self : Curlr Object
            DataFrame containing few columns, must have nb_visitors_id, nb_visitors_coh, cohort_size
            Index : List of id
        curlr_export_filename : string
            Full export_filename for saving the data
        current_wti : string
            Identity of the client throught the wti
        Returns
        -------
        DataFrame :
                columns : cohort_id(str) / qualif(str) / product_id(str)
                index : product_id (str)
        """
        tmp_c_url_r = self.ordering_columns()

        tmp_c_url_r = tmp_c_url_r.astype(dtype={'nb_visitors_id': int, 'nb_visitors_coh': int, 'cohort_size': int})

        tmp_c_url_r.to_csv(curlr_export_filename)

        tmp_c_url_r.reset_index(inplace=True)

        c_url_r_dict = CohortMatrix()
        c_url_r_dict.data = tmp_c_url_r.to_dict(orient='dict')
        c_url_r_dict.version = "1.0.0"
        c_url_r_dict.vars = {}
        c_url_r_dict.wti = current_wti
        c_url_r_dict = c_url_r_dict.to_dict()

        curlr_export_filename_json = curlr_export_filename[:-4]
        curlr_export_filename_json += ".json"

        with open(curlr_export_filename_json, 'w+', encoding='utf-8') as json_file:
            json.dump(c_url_r_dict, json_file, indent=4)
        json_file.close()

    def convert_product_id_into_cohort(self, product_id):

        cohort_id = None
        if product_id in self.index:
            cohort_id = self.loc[product_id, 'cohort_id']
        return cohort_id


class Durl(pd.DataFrame):

    def display_import_durl(self):
        print("Number of id : " + str(len(self.index)))
        print('\n', "Import D url - done", '\n')


class Luw(pd.DataFrame):

    def display_import_luw(self):
        print("Number of visits on product pages (only 1 by wvi and by url) : " + str(len(self)))
        print("Number of visitors on any product page : " + str(self['visitor_id'].nunique()))
        print("Number of different ids : " + str(self['product_id'].nunique()))
        print('\n', "Import L u,w - done", '\n')

    def display_import_info(self):
        print("nb actions in luw :", len(self))
        print("nb visitors in luw :", self['visitor_id'].nunique())
        print("nb product_id in luw :", self['product_id'].nunique())
        print("nb products seen by visitor :", len(self) / self['visitor_id'].nunique())

    def filter_product_ids_from_catalog(self, catalog):
        product_list = catalog.products_id_list
        self['product_id'] = self['product_id'].progress_apply(lambda x: x if x in product_list else np.NaN)
        self.dropna(inplace=True, subset=["product_id"])

    def filter_product_ids_from_list_of_ids(self, list_of_ids):
        if not isinstance(list_of_ids, list):
            raise TypeError(f"List_of_ids isn't a list type, but a {type(list_of_ids)} type")
        self['product_id'] = self['product_id'].progress_apply(lambda x: x if x not in list_of_ids else np.NaN)
        self.dropna(inplace=True, subset=["product_id"])

    def filter_product_ids_in_list_of_ids(self, list_of_ids):
        if not isinstance(list_of_ids, list):
            raise TypeError(f"List_of_ids isn't a list type, but a {type(list_of_ids)} type")
        self['product_id'] = self['product_id'].progress_apply(lambda x: x if x in list_of_ids else np.NaN)
        self.dropna(inplace=True, subset=["product_id"])

    def add_column_category(self, catalog_df):
        cat_list = []
        for i in range(len(self)):
            tmp_ref = None
            tmp_id = self['product_id'].iloc[i]
            if tmp_id in catalog_df.index:
                tmp_ref = catalog_df["category"].loc[tmp_id]
            cat_list.append(tmp_ref)
        self['category'] = cat_list


class Mvis:

    """
    matrice creuse
    index : product_ids
    columns : visitor_ids
    values : matrice creuse sans index ni columns (d'où la nécessité de les avoir par ailleurs) des visiteurs/ids
    m_vis : matrice utilisée pour le clustering, peut être dense, creuse

    Produit en ligne car on n'utilise plus de dense mais des matrices creuses.
    Creuse, accès aux lignes ou colonnes similaire, même si pour le format csr, les lignes sont accédés plus facilement
    """

    visitor_ids_list = []   # index
    product_ids_list = []   # columns
    values = None           # clustering matrix if not none
    m_vis = None            # sparse matrix

    def __init__(self, **kwargs):
        if 'visitor_ids_list' in kwargs:
            self.visitor_ids_list = kwargs['visitor_ids_list']
        if 'product_ids_list' in kwargs:
            self.product_ids_list = kwargs['product_ids_list']
        if 'values' in kwargs:
            self.values = kwargs['values']
        if 'm_vis' in kwargs:
            self.m_vis = kwargs['m_vis']

    def display_import_mvisf(self, is_test=False):

        if not is_test:
            nb_total_visits = np.sum(self.m_vis.nnz())
            print("Number of visits on product pages in array (only 1 by wvi and by id) : " + str(nb_total_visits))
            print("Number of visitors on any product page in array : " + str(len(self.visitor_ids_list)))
            print("Number of id analysed : " + str(len(self.product_ids_list)))
            print('\n', "Import Mvis - done", '\n')


class MvisDense(pd.DataFrame):
    """
    index : visitor_ids
    columns : product_ids
    """

    def rename_columns_to_int(self, dict_products_corresp_id_int):
        new_index = self.columns.copy()
        new_index = new_index.map(lambda x: dict_products_corresp_id_int[x])
        self.columns = new_index
        # print(new_index)

        self.sort_index(axis='columns', ascending=True, inplace=True)


class MvisThresholds:

    nb_visits_min_threshold_id = 0
    nb_visits_max_threshold_id = 0
    nb_visits_min_threshold_visitors = 0
    nb_visits_max_threshold_visitors = 0

    def __init__(self, **kwargs):
        if 'nb_visits_min_threshold_id' in kwargs:
            self.nb_visits_min_threshold_id = kwargs['nb_visits_min_threshold_id']
        if 'nb_visits_max_threshold_id' in kwargs:
            self.nb_visits_max_threshold_id = kwargs['nb_visits_max_threshold_id']
        if 'nb_visits_min_threshold_visitors' in kwargs:
            self.nb_visits_min_threshold_visitors = kwargs['nb_visits_min_threshold_visitors']
        if 'nb_visits_max_threshold_visitors' in kwargs:
            self.nb_visits_max_threshold_visitors = kwargs['nb_visits_max_threshold_visitors']

    def to_dict(self):
        return {'nb_visits_min_threshold_id': self.nb_visits_min_threshold_id,
                'nb_visits_max_threshold_id': self.nb_visits_max_threshold_id,
                'nb_visits_min_threshold_visitors': self.nb_visits_min_threshold_visitors,
                'nb_visits_max_threshold_visitors': self.nb_visits_max_threshold_visitors}


class Product:
    """
    :param id: Unique WAG Product ID
    :param ref: Unique Client Product ID
    :param url: url, string
    :param labels: dict. for example : recommendable True or False
    :param language: Language of the current URL
    :param price: Price in float
    :param created_at: creation of the product
    :param expired_at: Expiration of the product
    :param template: Template of the product
    :param nb_visitors : int
    :param keywords: Keywords
    """

    id = None
    ref = None
    url = None
    name = None
    labels = {}
    language = None
    price = None
    created_at = None
    expired_at = None
    template = None
    nb_visitors = None
    keywords = []

    def __init__(self, **kwargs):
        if 'id' in kwargs:
            self.id = kwargs['id']
        if 'ref' in kwargs:
            self.ref = kwargs['ref']
        if 'url' in kwargs:
            self.url = kwargs['url']
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'labels' in kwargs:
            self.labels = kwargs['labels']
        if 'language' in kwargs:
            self.language = kwargs['language']
        if 'price' in kwargs:
            self.price = kwargs['price']
        if 'created_at' in kwargs:
            self.created_at = kwargs['created_at']
        if 'expired_at' in kwargs:
            self.expired_at = kwargs['expired_at']
        if 'template' in kwargs:
            self.template = kwargs['template']
        if 'nb_visitors' in kwargs:
            self.nb_visitors = kwargs['nb_visitors']
        if 'keywords' in kwargs:
            self.keywords = kwargs['keywords']

    def __eq__(self, other):
        if isinstance(other, Product):
            return (self.id == other.id and self.ref == other.ref and self.url == other.url and
                    self.name == other.name and self.labels == other.labels and self.language == other.language and
                    self.price == other.price and self.created_at == other.created_at and self.expired_at == other.expired_at
                    and self.template == other.template and self.nb_visitors == other.nb_visitors
                    and self.keywords == other.keywords)
        # don't attempt to compare against unrelated types
        return NotImplemented

    def __hash__(self):
        # Can be defined because without nb_visitors, Product is an immutable object
        return hash((self.id, self.ref, self.url, self.name, self.price))

    def convert_into_category_azimut(self):
        ref = self.ref
        category = ref[2:4]
        return category

    def convert_into_cohort(self, curlr):

        cohort_id = None

        if self.id in curlr.index:
            cohort_id = curlr.loc[self.id, "cohort_id"]

        return cohort_id

    def to_dict(self):
        expired_at = self.expired_at
        if isinstance(expired_at, datetime):
            expired_at = self.expired_at

        return {
            'id': self.id,
            'ref': self.ref,
            'url': self.url,
            'name': self.name,
            'labels': self.labels,
            'language': self.language,
            'price': self.price,
            'expired_at': expired_at,
            'template': self.template,
            'nb_visitors': self.nb_visitors,
            'keywords': self.keywords,
        }

