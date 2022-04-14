import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.models import Luw
from src.src import convert_vect_into_ids, convert_id_into_category, split_path_and_last_product
from models.luw_manager import LuwManager
from models.mvisdense_manager import MvisDenseManager
from models.model_catalog import Catalog
from models.catalog_manager import CatalogManager

""" ******************************************************************************** """
""" TABLE DE CORRESPONDANCE                                                          """
""" ******************************************************************************** """

with open("data/tests_v1/dict_products_corresp_id_int_v1.pickle", 'rb') as file:
    dict_products_corresp_id_int = pickle.load(file)

with open("data/tests_v1/dict_products_corresp_int_id_v1.pickle", 'rb') as file:
    dict_products_corresp_int_id = pickle.load(file)

product_id_list = list(dict_products_corresp_int_id.values())
# print(dict_products_corresp_id_int, dict_products_corresp_int_id)

""" ******************************************************************************** """
""" VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
""" ******************************************************************************** """

luw = Luw(pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=None))
luw = Luw(luw[100000:].reset_index(drop=True))
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
""" CONTROLES DU MODELE                                                              """
""" ******************************************************************************** """

model = keras.models.load_model('data/tests_v1/model_azimut_v1')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(X)

#
# for i in [2, 3, 4]:
#     r = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
#     print("Visitor id : {}".format(visitors[i]))
#     print("Parcours : {}".format(r))
#     print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
#     print("Reality : {}".format(dict_products_corresp_int_id[y[i]]), '\n')


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


for i in range(len(X))[:30]:
    path = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
    cat_path_list = list(map(lambda x: convert_id_into_category(x, catalog), path))
    # print(cat_path_list)

    prediction = dict_products_corresp_int_id[np.argmax(predictions[i])]
    # print(prediction)

    cat_pred = convert_id_into_category(prediction, catalog)
    # print(cat_pred)

    print(cat_path_list, ' ---> ', cat_pred, '\n')

    # print("Visitor id : {}".format(visitors[i]))
    # print("Parcours : {}".format(path))
    # print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
    # print("Reality : {}".format(dict_products_corresp_int_id[y[i]]), '\n')