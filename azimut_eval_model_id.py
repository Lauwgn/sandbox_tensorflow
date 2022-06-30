import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.models import Luw
from src.src import convert_vect_into_ids, convert_id_into_category, search_max_occurences, split_path_and_last_product
from src.control_by_cohort import control_by_cohort
from models.curlr_manager import CurlrManager
from models.luw_manager import LuwManager
from models.mvisdense_manager import MvisDenseManager
from models.model_catalog import Catalog
from models.catalog_manager import CatalogManager

""" ******************************************************************************** """
""" IMPORT DU MODELE                                                                 """
""" ******************************************************************************** """

model = keras.models.load_model('data/tests_v1/model_azimut_v1')
# model = keras.models.load_model('data/tests_v2/model_azimut_v2')

""" ******************************************************************************** """
""" TABLE DE CORRESPONDANCE                                                          """
""" ******************************************************************************** """

with open("data/tests_v1/dict_products_corresp_id_int_v1.pickle", 'rb') as file:
# with open("data/tests_v2/dict_products_corresp_id_int.pickle", 'rb') as file:
    dict_products_corresp_id_int = pickle.load(file)

with open("data/tests_v1/dict_products_corresp_int_id_v1.pickle", 'rb') as file:
# with open("data/tests_v2/dict_products_corresp_int_id.pickle", 'rb') as file:
    dict_products_corresp_int_id = pickle.load(file)

""" ******************************************************************************** """
""" IMPORT LUW                                                                       """
""" ******************************************************************************** """

luw = Luw(pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=None))
luw = Luw(luw[100000:].reset_index(drop=True))


""" ******************************************************************************** """
""" CONTROLES PAR LA COHORTE                                                         """
""" ******************************************************************************** """
cohort = False
if cohort:
    curlr = CurlrManager.import_c_url_r('data/output_30_533d1d6652e1_210301-210905_curlr_revu_url_5-0_wvi_2-28.csv')
    print(curlr)

    control_by_cohort(model, dict_products_corresp_int_id, dict_products_corresp_id_int, luw, curlr)
    exit()


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
""" SPLIT : DATA // EXPECTED RESULT                                                  """
""" ******************************************************************************** """

visitors, luw_path, last_product_list = split_path_and_last_product(visits_min_df)
print("Check equality {}, {}, {}".format(len(visitors), luw_path.index.nunique(), len(last_product_list)), '\n')

""" ******************************************************************************** """
""" PREPARATION DES DONNEES                                                          """
""" ******************************************************************************** """

luw_path.reset_index(inplace=True)

mvis = MvisDenseManager.make_mvisdense(luw_path, product_id_list)
# print(type(mvis))

mvis.rename_columns_to_int(dict_products_corresp_id_int)
mvis = mvis.loc[visitors]
# print(mvis)

last_product_list_int = list(map(lambda x: dict_products_corresp_id_int[x], last_product_list))
print('\n', "Maximum des id dans last product seen : {}".format(np.max(last_product_list_int)))
print("Longueur de last product seen : {}".format(len(last_product_list_int)))
# print(last_product_list)
# print(expected_list_int)
X = mvis.values
y = np.array(last_product_list_int)

print("X.shape : {}".format(X.shape),
      "y.shape : {}".format(y.shape),
      "visitors.shape : {}".format(visitors.shape),
      '\n', sep='\n')

""" ******************************************************************************** """
""" TRAVAIL SUR LE MODELE                                                            """
""" ******************************************************************************** """

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X)

# print(predictions)

evaluation = model.evaluate(X, y)
print(evaluation)


evaluation_proba = probability_model.evaluate(X, y)
print(evaluation_proba)

exit()

""" ******************************************************************************** """
""" CONTROLES DU MODELE - PAR PRODUIT                                                """
""" ******************************************************************************** """

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(X)

# # for i in [2, 3, 4]:
# #     r = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
# #     print("Visitor id : {}".format(visitors[i]))
# #     print("Parcours : {}".format(r))
# #     print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
# #     print("Reality : {}".format(dict_products_corresp_int_id[y[i]]), '\n')
#
# # for i in range(X.shape[0]):
# # for i in range(5):
# #     print(np.argmax(predictions[i]), y[i])
#
# # print(dict_products_corresp_int_id)
# y_id = np.apply_along_axis(lambda x: dict_products_corresp_int_id[x[0]], 0, [y])    # je ne comprends pas tout mais ca marche
# predictions_id = [dict_products_corresp_int_id[np.argmax(predictions[i])] for i in range(len(predictions))]
# # print(y_id)
# # print(predictions_id)
#
# comparison_df = pd.DataFrame.from_dict({'y': y_id, 'predictions': predictions_id})
# comparison_df['good'] = comparison_df['y'] == comparison_df['predictions']
# # print(comparison_df)
# # print(comparison_df['good'].value_counts())


""" ******************************************************************************** """
""" CONTROLES PAR LA CATEGORIE                                                       """
""" ******************************************************************************** """

catalog = CatalogManager.import_from_json('data/20211206-catalog-533d1d6652e1-fr-en.json')
# print(catalog)
# print(catalog.products[0].ref)

# # CONTROLES SUR REFERENCES
# ref_list = []
# for tmp_prod in catalog.products:
#     ref_list.append(tmp_prod.convert_into_category_azimut())
# # print(ref_list)
#
# ref_df = pd.DataFrame(data=ref_list, columns=["ref"])
# print(ref_df)
# ref_occurences = ref_df.value_counts(subset=["ref"], sort=True, ascending=False)
# print(ref_occurences)


# for i in range(len(X))[:30]:
#     path = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
#     cat_path_list = list(map(lambda x: convert_id_into_category(x, catalog), path))
#     # print(cat_path_list)
#
#     category = search_max_occurences(path)
#     prediction = dict_products_corresp_int_id[np.argmax(predictions[i])]
#     # print(prediction)
#
#     cat_pred = convert_id_into_category(prediction, catalog)
#     # print(cat_pred)
#
#     print(cat_path_list, ' ---> ', category, ' ---> ', cat_pred, '\n')
#
#     # print("Visitor id : {}".format(visitors[i]))
#     # print("Parcours : {}".format(path))
#     # print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
#     # print("Reality : {}".format(dict_products_corresp_int_id[y[i]]), '\n')

# category_list, predictions_list = [], []
# for i in range(len(X)):
#     path = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
#     cat_path_list = list(map(lambda x: convert_id_into_category(x, catalog), path))
#     # print(cat_path_list)
#     category = search_max_occurences(cat_path_list)
#     category_list.append(category)
#
#     prediction = dict_products_corresp_int_id[np.argmax(predictions[i])]
#     # print(prediction)
#     cat_pred = convert_id_into_category(prediction, catalog)
#     predictions_list.append(cat_pred)
#
#
# category_df = pd.DataFrame.from_dict({"last_product": category_list, "cat_predict": predictions_list})
# category_df['good'] = category_df['last_product'] == category_df['cat_predict']
# print(category_df)
# print(category_df['good'].value_counts())





