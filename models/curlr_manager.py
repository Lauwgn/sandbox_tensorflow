import pandas as pd
import numpy as np
import unidecode
from collections import Counter
import json
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from models.models import Curlr
from models.catalog_manager import CatalogManager
# from src.tools.tools_wvi_path_df import determine_main_cohort_luw_sorted
# from src.tools.tools_text_treatment import *


class CurlrManager:

    @staticmethod
    def update_curlr_with_product_in_coh_minus_one(c_url_r_, new_products_id_list, catalog):

        c_url_r = c_url_r_.copy()
        c_url_r = c_url_r[['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']]

        tmp_qualif = c_url_r[c_url_r["cohort_id"] == "cohort_-1"]["qualif"].iloc[0]
        tmp_cohort_id = "cohort_-1"
        tmp_nb_visitors_id = 0
        tmp_nb_visitors_coh = 0
        tmp_cohort_size = 0
        tmp_url = ""
        tmp_name = ""

        new_rows_df = pd.DataFrame(index=pd.Index(data=[], name="id"),
                                   columns=c_url_r.columns)

        for tmp_id in new_products_id_list:
            tmp_product = CatalogManager.find_product_in_catalog_with_id(tmp_id, catalog)
            if tmp_product:
                if tmp_product.url:
                    tmp_url = tmp_product.url
                if tmp_product.name:
                    tmp_name = tmp_product.name

                new_row = pd.DataFrame(index=pd.Index(data=[tmp_id], name="id"),
                                       columns=c_url_r.columns,
                                       data=[[tmp_cohort_id, tmp_qualif, tmp_nb_visitors_id, tmp_nb_visitors_coh,
                                              tmp_cohort_size, tmp_url, tmp_name]])
                new_rows_df = new_rows_df.append(new_row)

        c_url_r = c_url_r.append(new_rows_df)

        return Curlr(c_url_r)

    @staticmethod
    def filter_list_with_only_ids_not_in_curlr(c_url_r, new_products_id_list):

        product_list = np.setdiff1d(new_products_id_list, c_url_r.index).tolist()

        return product_list

    @staticmethod
    def import_c_url_r(filename, nrows=None):
        """Import from a file all the data for a mvis object

        Parameters
        ----------
        filename : :string
            String containing at least an extension pickle or csv
        nrows : int
            Number of rows to take from the file, if a csv

        Returns
        -------
        DataFrame :
                columns : cohort_id(str) / qualif(str) / product_id(str)
                index : product_id (str)
        """

        c_url_r = pd.read_csv(filename, sep=',', nrows=nrows)  # Datafile
        c_url_r.rename(columns={"id": "product_id"},
                       inplace=True, errors="ignore")
        c_url_r.set_index(keys='product_id', drop=True, inplace=True)
        c_url_r = Curlr(c_url_r)

        c_url_r = c_url_r.change_types()
        print(f"Check type c_url_r : {c_url_r.check_types()}")
        c_url_r.display_import_curlr()
        return c_url_r

    @staticmethod
    def import_c_url_r_json(filename):

        with open(filename, 'r+') as json_file:
            data_dict = json.load(json_file)

        c_url_r = Curlr(pd.DataFrame.from_dict(data_dict['data']))
        c_url_r.set_index(keys='id', drop=True, inplace=True)

        c_url_r.display_import_curlr()

        return c_url_r



    @staticmethod
    def add_price_column(c_url_r, catalog, is_test=False):

        ids = c_url_r.index.tolist()
        products_ids = catalog.products_id_list

        list_ind_catalog = [products_ids.index(str(id_temp)) if str(id_temp) in products_ids else None for id_temp in ids]
        price_list = [catalog.select_product_by_indice(indice_url).price if indice_url is not None else None for indice_url
                      in list_ind_catalog]
        price_list = [np.nan if price is None else float(price) for price in price_list]

        c_url_r['price'] = price_list

        if not is_test:
            print("Add Price Done !")

        return c_url_r

    # @staticmethod
    # def add_keywords_freq(c_url_r, from_file=False, is_test=False):
    #     # Require the nan module from the numpy library
    #
    #     if "keywords" not in c_url_r.columns:
    #         print("Column keywords not in c_url")
    #         return c_url_r
    #     else:
    #         """ Si provient d'un fichier, utiliser eval(str(x))"""
    #         if from_file:
    #             c_url_r['keywords'] = c_url_r['keywords'].apply(lambda x: eval(str(x)))
    #
    #         keywords_df = c_url_r.groupby(by=['cohort_id'])['keywords'].apply(list)
    #         keywords_df = keywords_df.apply(lambda x: flatten_list(x, flat_list=[]))
    #
    #         # print("Nombre de mots clés au total : {}".format(len(set(flatten_list(keywords_clean_df.tolist(), flat_list)))))
    #         keywords_df = keywords_df.apply(lambda x: [word.lower() for word in x if type(word) == str])
    #
    #         keywords_df = keywords_df.apply(lambda x: Counter(x).most_common(15))
    #         keywords_df.rename("keywords_freq")
    #         # new_c_url_r = c_url_r.merge(keywords_df, how='inner', on="cohort_id", left_index=True)
    #         new_c_url_r = c_url_r.merge(keywords_df, how='inner', on="cohort_id", suffixes=('', '_freq'))
    #
    #         new_c_url_r.index = c_url_r.index
    #
    #         if not is_test:
    #             print("Add Keywords Freq Done !")
    #
    #     return new_c_url_r

    @staticmethod
    def add_keywords_column(c_url_r, catalog, is_test=False):

        ids = c_url_r.index.tolist()
        products_ids = catalog.products_id_list
        list_ind_catalog = [products_ids.index(str(id_temp)) if str(id_temp) in products_ids else "" for id_temp in ids]
        key_list = [catalog.select_product_by_indice(indice_url).keywords if indice_url != "" else "" for indice_url in
                    list_ind_catalog]

        c_url_r['keywords'] = key_list

        if not is_test:
            print("Add Keywords Done !")

        return c_url_r

    # @staticmethod
    # def add_template_freq(c_url_r, from_file=False, is_test=False):
    #     # Require the nan module from the numpy library
    #
    #     """ Si provient d'un fichier, utiliser eval(str(x))"""
    #     if from_file:
    #         c_url_r['template'] = c_url_r['template'].apply(lambda x: eval(str(x)))
    #
    #     template_df = c_url_r.groupby(by=['cohort_id'])['template'].apply(list)
    #     template_df = template_df.apply(lambda x: flatten_list(x, flat_list=[]))
    #
    #     # print("Nombre de mots clés au total : {}".format(len(set(flatten_list(keywords_clean_df.tolist(), flat_list)))))
    #     template_df = template_df.apply(lambda x: [word.lower() for word in x if type(word) == str])
    #
    #     template_df = template_df.apply(lambda x: Counter(x).most_common(15))
    #     template_df.rename("template_counter")
    #
    #     new_c_url_r = c_url_r.merge(template_df, how='inner', on="cohort_id", suffixes=('', '_freq'))
    #
    #     new_c_url_r.index = c_url_r.index
    #
    #     if not is_test:
    #         print("Add Template Freq Done !")
    #
    #     return new_c_url_r

    @staticmethod
    def add_template_column(c_url_r, catalog, is_test=False):
        # ids and product_ids both in str
        ids = c_url_r.index.tolist()
        products_ids = catalog.products_id_list

        list_ind_catalog = [products_ids.index(str(id_temp)) if str(id_temp) in products_ids else "" for id_temp in ids]
        template_list = [catalog.select_product_by_indice(indice_url).template if indice_url != "" else "" for
                         indice_url in list_ind_catalog]

        c_url_r['template'] = template_list

        if not is_test:
            print("Add Template Done !")

        return c_url_r

    @staticmethod
    def add_url_column_id(c_url_r, catalog, is_test=False):

        ids = c_url_r.index.tolist()
        products_ids = catalog.products_id_list

        list_ind_catalog = [products_ids.index(str(id_temp)) if str(id_temp) in products_ids else "" for id_temp in ids]
        url_list = [catalog.select_product_by_indice(indice_url).url if indice_url != "" else "" for indice_url in
                    list_ind_catalog]
        c_url_r['url'] = url_list

        if not is_test:
            print("Add Url Done !")

        return c_url_r

    @staticmethod
    def add_url_column_ref(c_url_r, catalog, is_test=False):

        refs = c_url_r.index.tolist()
        products_refs = [product.ref for product in catalog.products]

        list_ref_catalog = [products_refs.index(str(id_temp)) if str(id_temp) in products_refs else "" for id_temp in
                            refs]
        url_list = [catalog.select_product_by_indice(indice_url).url if indice_url != "" else "" for indice_url in
                    list_ref_catalog]

        c_url_r['url'] = url_list

        if not is_test:
            print("Add Url Done !")

        return c_url_r

    # @staticmethod
    # def add_name_freq(c_url_r, from_file=False, is_test=False):
    #     # Require the nan module from the numpy library
    #     #
    #     """ Si provient d'un fichier, utiliser eval(str(x))"""
    #     if from_file:
    #         c_url_r['name'] = c_url_r['name'].apply(lambda x: eval(str(x)))
    #     # On coupe la description en mots
    #     c_url_r['name_cleaned'] = c_url_r['name'].apply(lambda x: cut_list(x, cutted_list=[]))
    #     # On enlève les majuscules
    #     c_url_r['name_cleaned'] = c_url_r['name_cleaned'].apply(
    #         lambda x: [word.lower() for word in x if type(word) == str])
    #     # On enlève les mots ayant peu de sens (voir liste dans la fonction stop_words)
    #     c_url_r['name_cleaned'] = c_url_r['name_cleaned'].apply(lambda x: stop_words(x))
    #     # On enlève les accents
    #     c_url_r['name_cleaned'] = c_url_r['name_cleaned'].apply(lambda x: [unidecode.unidecode(string) for string in x])
    #     # On enlève les mots de longueur 1
    #     c_url_r['name_cleaned'] = c_url_r['name_cleaned'].apply(lambda x: [string for string in x if len(string) > 1])
    #     # On concatène les descriptions de chaque produits par cohorte
    #     name_df = c_url_r.groupby(by=['cohort_id'])['name_cleaned'].apply(list)
    #     # On applati les listes de listes de chaque cohortes
    #     name_df = name_df.apply(lambda x: flatten_list(x, flat_list=[]))
    #     # On enlève les mots au pluriels
    #     name_df = name_df.apply(lambda x: remove_plurals(x, no_plural=[]))
    #     # On ne conserve que les 15 premiers élements les plus redondants
    #     name_df = name_df.apply(lambda x: Counter(x).most_common(15))
    #
    #     # On récupère
    #     new_c_url_r = c_url_r.merge(name_df, how='inner', on="cohort_id", suffixes=('', '_freq'))
    #
    #     new_c_url_r.index = c_url_r.index
    #
    #     if not is_test:
    #         print("Add Name Freq Done !")
    #
    #     return new_c_url_r

    @staticmethod
    def add_name_column_id(c_url_r, catalog, is_test=False):

        ids = c_url_r.index.tolist()
        products_ids = catalog.products_id_list

        list_ind_catalog = [products_ids.index(str(id_temp)) if str(id_temp) in products_ids else "" for id_temp in ids]
        name_list = [catalog.select_product_by_indice(indice_url).name if indice_url != "" else "" for indice_url in
                     list_ind_catalog]

        c_url_r['name'] = name_list

        if not is_test:
            print("Add Name Done !")

        return c_url_r

    @staticmethod
    def add_ranking_columns(c_url_r, mvis_obj, epochs=5, nombre_reco=4, verbose=False):
        """Create a ranking of recommandation for a c_url_r

        Parameters
        ----------
        c_url_r : Curlr
            Dataframe containing a column cohort_id and nb_visitors_id, with those two columns
        mvis_obj : Mvis
            Number of rows to take from the file, if a csv
        epochs : int
            Each epoch find the most seen viewed product and give the ranks of his nearest neighbors
        nombre_reco : int
            Number of nearest neighbors to find for each epochs
        verbose : bool
            Show process
        Returns
        -------
        DataFrame :
                columns : cohort_id(str) / qualif(str) / product_id(str)
                index : product_id (str)
        """
        mvis = mvis_obj.values
        list_ids = mvis_obj.product_ids_list
        ids_not_minus_1 = c_url_r[c_url_r['cohort_id'] != 'cohort_-1'].index.tolist()
        ids_minus_1 = c_url_r[c_url_r['cohort_id'] == 'cohort_-1'].index.tolist()

        # Ranking de notre curlr pour les produits en dehors de la -1
        c_url_r.loc[ids_not_minus_1, ["ranking"]] = \
            c_url_r[c_url_r['cohort_id'] != 'cohort_-1'].groupby('cohort_id')['nb_visitors_id'].\
                rank(method="first", ascending=False)

        # Ranking de notre curlr pour les produits de la -1
        candidates = ids_minus_1
        if candidates:      # Si on a au moins 1 élement dans la -1

            for n_iter in range(epochs):
                if not candidates or len(candidates) == 1:
                    # Si on a aucun élément ou un seul dans les candidats potentiels de la -1
                    break
                curr_id = c_url_r.loc[candidates, ['nb_visitors_id']].idxmax().values[0]

                # On attribut le ranking à notre produit
                c_url_r.loc[[curr_id], ["ranking"]] = (nombre_reco+1)*n_iter + 1

                if curr_id not in list_ids:     # On assigne un rang et on passe
                    continue

                d_url_min_1 = pd.DataFrame(index=candidates, columns=[curr_id])
                #todo : change d_url boucle

                if issparse(mvis):
                    """ On recupère la liste des indices du mvis de la nouvelle liste d"élements de la -1"""
                    tmp_id_list_1, _, ind_id_list_min_1 = np.intersect1d(candidates, list_ids, return_indices=True)

                    d_url_min_1[curr_id] = pairwise_distances(mvis[ind_id_list_min_1, :],
                                                              mvis[list_ids.index(curr_id), :],
                                                              metric="euclidean")
                else:
                    d_url_min_1[curr_id] = pairwise_distances(mvis.loc[candidates, :].to_numpy(),
                                                              mvis.loc[curr_id, :].to_numpy().reshape(1, -1),
                                                              metric="euclidean")

                # On attribut le ranking aux produits les plus proches de celui choisi
                neighbors_ids = d_url_min_1.nsmallest(n=nombre_reco, columns=curr_id, keep="all").index.tolist()

                try:
                    neighbors_ids.remove(curr_id)   # Le produit le plus proche contient forcément lui-même
                except ValueError:
                    if verbose:
                        print(f"Produit {curr_id} not in {neighbors_ids}")
                if verbose:
                    print(f"Neighbors_id {neighbors_ids}")

                if n_iter == 0:
                    n_recommended = 2
                    old_n_recommended = n_recommended + len(neighbors_ids)
                else:
                    n_recommended = old_n_recommended
                    old_n_recommended = n_recommended + len(neighbors_ids)

                if verbose:
                    print(f"n_recommended {n_recommended}")
                    print(f"old_n_recommended {old_n_recommended}")

                c_url_r.loc[neighbors_ids, ["ranking"]] = range(n_recommended, old_n_recommended)

                if verbose:
                    print(f"candidates before removing id {candidates}")
                    print(f"curr_id {curr_id}")

                candidates.remove(curr_id)

                if verbose:
                    print(f"candidates after removing id {candidates}")
                for neighbors in neighbors_ids:
                    if verbose:
                        print(f"neighbors to remove : {neighbors}")
                    candidates.remove(neighbors)

                n_iter += 1

            if verbose:
                print(f"candidates after epochs {candidates}")
                print(f"n_recommended after epochs {n_recommended}")
                print(f"old_n_recommended after epochs {old_n_recommended}")

            c_url_r.loc[candidates, ["ranking"]] = c_url_r.loc[candidates, :]['nb_visitors_id'].\
                rank(method="first", ascending=False)

            c_url_r.loc[candidates, ["ranking"]] += old_n_recommended - 1
        if not verbose:
            print("Add Rankings Done !")

        return c_url_r

    @staticmethod
    def add_nb_wvi_by_id_sparse(c_url_r, m_vis, list_ids, is_test=False):

        id_list, ind_list_ids, ind_c_url = np.intersect1d(list_ids, c_url_r.index.values, return_indices=True)

        nb_wvi_list = [m_vis[tmp_ind_id, :].sum() for tmp_ind_id in ind_list_ids]
        nb_wvi_series = pd.Series(index=id_list, data=nb_wvi_list)

        c_url_r['nb_visitors_id'] = nb_wvi_series

        c_url_r.fillna(value=0, inplace=True)

        c_url_r['nb_visitors_id'] = pd.to_numeric(c_url_r['nb_visitors_id'])

        if not is_test:
            print("Nb wvi par id Done !")

        return c_url_r

    @staticmethod
    def return_nb_id_coh(nb_id_series, current_cohort_):

        if current_cohort_ in nb_id_series.index:
            return nb_id_series.loc[current_cohort_]
        else:
            return 0

    @staticmethod
    def return_nb_visitors_coh(nb_wvi_series_, current_cohort_):

        if current_cohort_ in nb_wvi_series_.index:
            return nb_wvi_series_.loc[current_cohort_]
        else:
            return 0

    # @staticmethod
    # def add_nb_wvi_by_cohort_sparse(c_url_r=Curlr(), mvis_obj=None, is_test=False):
    #     """Import from a file all the data for a mvis object
    #
    #     Parameters
    #     ----------
    #     c_url_r : Curlr
    #         Curlr object contenant les en id les id produits et la colonne cohort_id
    #     mvis_obj : Mvis
    #         Objet mvis, doit avoir les product_id en lignes et les visitor_id en colonnes
    #     Returns
    #     -------
    #     Curlr :
    #             columns : base + nb_visitors_coh
    #             index : product_id (str)
    #     """
    #
    #     list_wvi = mvis_obj.visitor_ids_list
    #     list_ids = mvis_obj.product_ids_list
    #     tmp_mvis = mvis_obj.m_vis  # Permet accès au parcours d'un wvi
    #     dict_id_to_cohort = dict(zip(c_url_r.index, c_url_r['cohort_id']))
    #     dict_cohort_nbwvi = dict(zip(c_url_r['cohort_id'].unique(), [0]*c_url_r['cohort_id'].nunique()))
    #     if "cohort_-1" not in dict_cohort_nbwvi.keys():
    #         dict_cohort_nbwvi["cohort_-1"] = 0
    #     # Pour chaque visiteur, on va determiner sa cohorte associé
    #     for count, current_wvi in enumerate(list_wvi):
    #         # On récupère les colonnes, puis on prends la ligne correspondante
    #         list_ind_ids = tmp_mvis.getcol(count).nonzero()[0]
    #         wvi_path = [list_ids[curr_ind] for curr_ind in list_ind_ids]  # liste des ids
    #         current_wvi_cohort = determine_main_cohort_luw_sorted(wvi_path=wvi_path,
    #                                                               dict_id_to_cohort=dict_id_to_cohort,
    #                                                               include_minus_1=True)
    #
    #         dict_cohort_nbwvi[current_wvi_cohort] += 1
    #     c_url_r['nb_visitors_coh'] = c_url_r.apply(lambda x: dict_cohort_nbwvi[x.cohort_id], axis=1)
    #
    #     if not is_test:
    #         print("Nb wvi par cohort Done !")
    #
    #     return c_url_r

    @staticmethod
    def add_name_column_ref(c_url_r, catalog, is_test=False):

        refs = c_url_r.index.tolist()
        products_refs = [product.ref for product in catalog.products]

        list_ind_catalog = [products_refs.index(str(id_temp)) if str(id_temp) in products_refs else "" for id_temp in
                            refs]
        name_list = [catalog.select_product_by_indice(indice_url).name if indice_url != "" else "" for indice_url in
                     list_ind_catalog]

        c_url_r['name'] = name_list

        if not is_test:
            print("Add Name Done !")

        return c_url_r

    @staticmethod
    def add_visitors_stats_to_curlr_sparse(c_url_r, mvis_obj, is_test=False):
        # En import : mvis_obj --> Class Mvis, matrice creuse
        m_vis = mvis_obj.m_vis
        list_ids = mvis_obj.product_ids_list
        list_wvi = mvis_obj.visitor_ids_list

        """ Update nb visitor by id    """
        c_url_r = CurlrManager.add_nb_wvi_by_id_sparse(c_url_r, m_vis, list_ids=list_ids, is_test=is_test)

        """ Update rankings by cohort and id """
        c_url_r = CurlrManager.add_ranking_columns(c_url_r=c_url_r, mvis_obj=mvis_obj, epochs=5, verbose=is_test)

        """ Update nb visitor by cohort    """
        c_url_r = CurlrManager.add_nb_wvi_by_cohort_sparse(c_url_r=c_url_r, mvis_obj=mvis_obj, is_test=is_test)
        """ Update cohort_size - nb id by cohort """
        Curlr.update_cohort_size(c_url_r)

        c_url_r.sort_values(by=['nb_visitors_coh', 'cohort_id', 'nb_visitors_id'], inplace=True,
                            ascending=[False, True, False])

        return c_url_r

    @staticmethod
    def convert_ref_into_id_curlr(c_url_r, catalog):

        id_list = []

        for tmp_ref in c_url_r.index:
            tmp_prod = CatalogManager.find_product_in_catalog_with_ref(current_ref=tmp_ref, catalog=catalog)
            tmp_id = CatalogManager.get_attr_from_prod(current_product=tmp_prod, attribut="id")
            id_list.append(tmp_id)

        c_url_r.reset_index(inplace=True)
        c_url_r['product_id'] = id_list

        c_url_r.set_index(keys='product_id', inplace=True, drop=True)

        return c_url_r.copy()
