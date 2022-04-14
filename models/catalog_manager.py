import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from models.models import Product
from models.model_catalog import Catalog


class CatalogManager:
    """ Pratiques pour ajouter une nouvelle fonction
        Static_method
            Ajouter les parametres de fonctions à leurs appels, positionnements amenés à changer
        """
    @staticmethod
    def import_from_json(filename):
        with open(filename, 'r') as jsonfile:
            dict_catalog = json.load(jsonfile)

        products_list = []
        for prod in dict_catalog['products']:
            tmp_product = Product(**prod)

            products_list.append(tmp_product)

        tmp_updated_at, tmp_wti = None, None
        if 'updated_at' in dict_catalog.keys():
            tmp_updated_at = dict_catalog['updated_at']
        if 'wti' in dict_catalog.keys():
            tmp_wti = dict_catalog['wti']

        catalog = Catalog(products=products_list, updated_at=tmp_updated_at, wti=tmp_wti)
        catalog.generate_id_list()

        print("Import Catalog - Done ")

        return catalog

    @staticmethod
    def import_catalog_from_dict(dictionnary):
        # dictionnary : nested with "products" as first items, then a list of products
        # to access a product, dictionnary["products"][integer]
        products_list = []
        try:
            keys_dict = dictionnary['products'][0].keys()
        except:
            raise ValueError("Dictionnary doesn't have the format needed or is empty")

        available_dict_keys = []
        for key in keys_dict:
            if hasattr(Product, key):
                available_dict_keys.append(key)

        for product in dictionnary['products']:
            tmp_product = Product()
            for key in available_dict_keys:
                setattr(tmp_product, key, product[key])
            products_list.append(tmp_product)

        catalog = Catalog(products=products_list)
        catalog.generate_id_list()

        return catalog

    @staticmethod
    def import_catalog_json_with_ref(filename):
        with open(filename, 'r') as jsonfile:
            dict = json.load(jsonfile)

        products_list = []
        for prod in dict['products']:
            tmp_product = Product(id=prod['ref'], url=prod['url'])
            products_list.append(tmp_product)

        catalog = Catalog(products=products_list)
        catalog.generate_id_list()

        return catalog

    @staticmethod
    def import_catalog_json_with_id(filename):      # @todo : A SUPPRIMER ??
        with open(filename, 'r') as jsonfile:
            dict = json.load(jsonfile)

        products_list = []
        for prod in dict['products']:
            tmp_station, tmp_niveau = None, None

            if 'extra' in prod.keys():
                tmp_extra = prod['extra']
                if 'group_name' in tmp_extra.keys():
                    tmp_station = tmp_extra['group_name']
                if 'physical_difficulty' in tmp_extra.keys():
                    tmp_niveau = tmp_extra['physical_difficulty']

            tmp_product = Product(id=prod['id'], url=prod['url'], station=tmp_station, niveau=tmp_niveau)
            products_list.append(tmp_product)

        catalog = Catalog(products=products_list)
        catalog.generate_id_list()

        return catalog

    @staticmethod
    def is_product_in_catalog(product, catalog):
        """ Check if a product is in the catalog"""

        for curr_product in catalog.products:
            if product == curr_product:
                return True
        return False

    @staticmethod
    def is_id_in_catalog(current_id, catalog):
        """ Check if an id is in the catalog"""
        if current_id in catalog.products_id_list:
            return True
        return False

    @staticmethod
    def get_attr_from_prod(current_product, attribut=""):
        if current_product is None:
            return None

        if not isinstance(current_product, Product):
            raise TypeError("Type of product is not Product but {}", type(current_product))

        if hasattr(Product, attribut):
            return getattr(current_product, attribut)

    @staticmethod
    def fuse_catalog(catalog, catalog_filename):
        """ Check if a product is in the catalog"""

        catalog_old = CatalogManager.import_catalog_json_with_id(catalog_filename)
        catalog_old.display_import_info()

        adding_to_catalog = []
        count = 0
        list_id_to_add_from_old_catalog = list(set(catalog_old.products_id_list) - set(catalog.products_id_list))
        for curr_prod in catalog_old.products:
            if curr_prod.id in list_id_to_add_from_old_catalog:
                count += 1
                adding_to_catalog.append(curr_prod)

        # print(f"Nombre d'ids à enlever {count}")
        # print(f"Nombre d'ids dans la liste {len(list_id_to_add_from_old_catalog)}")
        catalog.products.extend(adding_to_catalog)

        catalog.generate_id_list()
        catalog.display_import_info()

        return catalog

    @staticmethod
    def filter_list_with_only_ids_in_catalog(current_id_list, catalog):
        catalog_id_list = catalog.products_id_list
        return np.intersect1d(current_id_list, catalog_id_list).tolist()

    @staticmethod
    def find_product_in_catalog_with_attributs(catalog, attribut="", attr_value="", verbose=False):
        # Function to use only when the attribut requested is unique per product
        current_product = None

        for product in catalog.products:
            if getattr(product, attribut) == attr_value:
                current_product = product
                break

        if current_product is None:
            if verbose:
                print(f"Aucun attribut {attribut} ne contient la valeur {attr_value} dans le catalog")
        return current_product

    @staticmethod
    def find_product_in_catalog_with_id(current_id, catalog):

        found_in_catalog = False
        current_product = None
        products_list = catalog.products
        i = 0

        while not found_in_catalog and i < len(products_list):
            if str(products_list[i].id) == str(current_id):
                current_product = products_list[i]
                found_in_catalog = True
            i += 1

        return current_product

    @staticmethod
    def find_product_in_catalog_with_ref(current_ref, catalog):

        found_in_catalog = False
        current_product = None
        products_list = catalog.products
        i = 0

        while not found_in_catalog and i < len(products_list):
            if str(products_list[i].ref) == str(current_ref):
                current_product = products_list[i]
                found_in_catalog = True
            i += 1

        return current_product

    @staticmethod
    def transform_catalog_into_series(catalog):
        id_list, url_list = [], []
        for tmp_product in catalog.products:
            id_list.append(getattr(tmp_product, 'id'))
            url_list.append(getattr(tmp_product, 'url'))
        catalog_df = pd.DataFrame.from_dict({'id': id_list, 'url': url_list})
        catalog_df.set_index(keys=['url'], inplace=True)
        catalog_series = catalog_df['id']

        return catalog_series

    @staticmethod
    def select_specific_ids_from_catalog(catalog, current_wti):
        if current_wti == '55ee9f613ece':
            catalog = CatalogManager.select_catalog_for_crt_normandie(catalog=catalog)
        elif current_wti == 'ce891b2afd0a':
            catalog = CatalogManager.select_catalog_for_grand_bornand_ot(catalog=catalog)
        elif current_wti == "5616a877e3e3":   #CRT PACA
            catalog = CatalogManager.select_catalog_for_crt_paca(catalog=catalog)
        else:
            catalog = catalog

        return catalog

    @staticmethod
    def remove_product_with_bad_url(catalog=Catalog, accepted_url_list=[], to_remove=[]):

        products_to_remove = []
        for tmp_prod in catalog.products:
            if Catalog.bad_url(x=tmp_prod.url, to_remove=to_remove, to_keep=accepted_url_list):
                products_to_remove.append(tmp_prod)
        # print(f"Nombre d_urls mauvaises {len(products_to_remove)}")

        products_to_remove = list(set(products_to_remove))  # Empêche les doublons
        for product in products_to_remove:
            catalog.remove_product(product=product)
        catalog.generate_id_list()

        # print(f"Nombre produits catalog finaux {len(catalog.products_id_list)}")

        return catalog

    @staticmethod
    def remove_product_with_bad_template(catalog=Catalog, accepted_templates=[]):

        products_to_remove = []
        for tmp_prod in catalog.products:
            if tmp_prod.template not in accepted_templates:
                products_to_remove.append(tmp_prod)
        # print(f"Nombre de produits enlevés {len(products_to_remove)}")

        products_to_remove = list(set(products_to_remove))  # Empêche les doublons
        for product in products_to_remove:
            catalog.remove_product(product=product)
        catalog.generate_id_list()

        # print(f"Nombre produits catalog finaux {len(catalog.products_id_list)}")

        return catalog


