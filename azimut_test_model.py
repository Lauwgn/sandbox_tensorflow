import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.src import convert_vect_into_ids, make_mvis, mvis_rename_columns, select_visitors_enough_visits, split_path_and_last_product
""" ******************************************************************************** """
""" TABLE DE CORRESPONDANCE                                                          """
""" ******************************************************************************** """

with open("data/dict_products_corresp_id_int.pickle", 'rb') as file:
    dict_products_corresp_id_int = pickle.load(file)

with open("data/dict_products_corresp_int_id.pickle", 'rb') as file:
    dict_products_corresp_int_id = pickle.load(file)

product_id_list = list(dict_products_corresp_int_id.values())
# print(dict_products_corresp_id_int, dict_products_corresp_int_id)

""" ******************************************************************************** """
""" VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
""" ******************************************************************************** """

luw = pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=None)

luw = luw[100000:]

# @todo : rajouter un filtre sur les produits de luw qui ne sont pas dans dict_products

visits_min_df = select_visitors_enough_visits(luw, 3, 10)




""" ******************************************************************************** """
""" SPLIT : DATA // EXPECTED RESULT                                                  """
""" ******************************************************************************** """

visitors, luw_path, last_product_list = split_path_and_last_product(visits_min_df)
print("Check equality {}, {}, {}".format(len(visitors), luw_path.index.nunique(), len(last_product_list)), '\n')

""" ******************************************************************************** """
""" PREPARATION DES DONNEES                                                          """
""" ******************************************************************************** """

luw_path.reset_index(inplace=True)

mvis = make_mvis(luw_path, product_id_list)
mvis = mvis_rename_columns(mvis, dict_products_corresp_id_int)
mvis = mvis.loc[visitors]
# print(mvis)


last_product_list_int = list(map(lambda x: dict_products_corresp_id_int[x], last_product_list))
print("Maximum des id dans last product seen : {}".format(np.max(last_product_list_int)))
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

model = keras.models.load_model('data/model_azimut')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(X)

for i in [2, 3, 4]:
    r = convert_vect_into_ids(X[i], dict_products_corresp_int_id)
    print("Visitor id : {}".format(visitors[i]))
    print("Parcours : {}".format(r))
    print("Prediction : {}".format(dict_products_corresp_int_id[np.argmax(predictions[i])]))
    print("Reality : {}".format(dict_products_corresp_int_id[y[i]]), '\n')
