import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import pickle
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras

from models.models import Luw
from src.src import split_path_and_last_product
from models.luw_manager import LuwManager
from models.mvisdense_manager import MvisDenseManager
from models.model_catalog import Catalog
from models.catalog_manager import CatalogManager


""" ******************************************************************************** """
""" IMPORT DU MODELE                                                                 """
""" ******************************************************************************** """

model = keras.models.load_model('data/models_category/model_azimut_category_prediction.h5')


""" ******************************************************************************** """
""" TABLES DE CORRESPONDANCE                                                          """
""" ******************************************************************************** """

with open("data/models_category/dict_products_corresp_id_int.pickle", 'rb') as file:
    dict_products_corresp_id_int = pickle.load(file)

with open("data/models_category/dict_products_corresp_int_id.pickle", 'rb') as file:
    dict_products_corresp_int_id = pickle.load(file)

with open("data/models_category/dict_products_corresp_cat_int.pickle", 'rb') as file:
    dict_products_corresp_cat_int = pickle.load(file)

with open("data/models_category/dict_products_corresp_int_cat.pickle", 'rb') as file:
    dict_products_corresp_int_cat = pickle.load(file)

""" ******************************************************************************** """
""" IMPORT LUW AND CATALOG                                                           """
""" ******************************************************************************** """

luw = Luw(pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=None))
luw = Luw(luw[100000:].reset_index(drop=True))

catalog = CatalogManager.import_from_json("data/20211206-catalog-533d1d6652e1-fr-en.json")
luw.filter_product_ids_from_catalog(catalog)
# print(luw)

catalog_df = pd.read_csv("data/catalog_azimut_cat_revu.csv")
catalog_df.set_index(keys="product_id", inplace=True)
# print(catalog_df)

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
""" PREPROCESSING                                                                    """
""" ******************************************************************************** """

luw_path.reset_index(inplace=True)
mvis_input = MvisDenseManager.make_mvisdense(luw_path, product_id_list)
mvis_input.rename_columns_to_int(dict_products_corresp_id_int)
mvis_input = mvis_input.loc[visitors]
x_test = mvis_input.values

expected_list_category = list(map(lambda x: catalog_df["category"].loc[x], last_product_list))
expected_list_category_int = list(map(lambda x: dict_products_corresp_cat_int[x], expected_list_category))
y_true = np.array(expected_list_category_int)

""" ******************************************************************************** """
""" TRAVAIL SUR LE MODELE                                                            """
""" ******************************************************************************** """

y_pred_sparse = model.predict(x_test)
y_pred = list(map(lambda x: np.argmax(x), y_pred_sparse))
# print(y_pred)
# print(y_true)
print("Longueur de y_pred : {}, Longueur de y_true : {}".format(len(y_pred), len(y_true)))

scce = tf.keras.losses.SparseCategoricalCrossentropy()
print("Sparse Categorical Cross Entropy y_true, y_pred : {}".format(scce(y_true, y_pred_sparse).numpy()))

acc = tf.keras.metrics.Accuracy()
print("Accuracy y_true, y_pred : {}".format(acc(y_true, y_pred).numpy()))

cm = confusion_matrix(y_true, y_pred, labels=np.arange(0, len(dict_products_corresp_int_cat)))
print('\n', cm)
print(dict_products_corresp_int_cat)

""" ******************************************************************************** """
""" PLOT : MATRICE DE CONFUSION                                                      """
""" ******************************************************************************** """

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.tight_layout()
plt.colorbar()

classes = [dict_products_corresp_int_cat[i] for i in range(len(dict_products_corresp_int_cat))]

# tick_marks, classes = np.arange(cm.shape[0]), np.arange(cm.shape[0])
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)

plt.xlabel('Predicted label')
plt.ylabel('True label')

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.show()

exit()

