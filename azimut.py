import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


luw = pd.read_csv('data/20220311-luw-533d1d6652e1-20210101-20220310.csv', nrows=50000)
# print(luw)

""" ******************************************************************************** """
""" VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
""" ******************************************************************************** """

nb_visits_series = luw['visitor_id'].value_counts().sort_values(ascending=False)
nb_visits_series = nb_visits_series[nb_visits_series <= 30]
# print(nb_visits_series)

visitor_id_3_visits_list = nb_visits_series[nb_visits_series >= 3].index.tolist()
visitor_id_4_visits_list = nb_visits_series[nb_visits_series >= 4].index.tolist()
# print(visitor_id_3_visits_list)
print("Nb de visiteurs avec 3 visites ou plus : {}".format(len(visitor_id_3_visits_list)),
      "Nb de visiteurs avec 4 visites ou plus : {}".format(len(visitor_id_4_visits_list)),
      sep='\n')

visits_3min_df = luw.set_index(keys=['visitor_id']).loc[visitor_id_3_visits_list][['product_id']].copy()
# print('\n', visits_3min_df)

""" ******************************************************************************** """
""" PREPARATION DES DONNEES                                                          """
""" ******************************************************************************** """

input_list, input_index, input_df_list, input_df_index, expected_list = [], [], [], [], []

# for tmp_visitor in visits_3min_df.index.unique()[:2]:
for tmp_visitor in visits_3min_df.index.unique():
    tmp_luw = visits_3min_df.loc[tmp_visitor]
    # print(tmp_luw)

    input_index.append(tmp_visitor)
    input_list.append(tmp_luw['product_id'][:-1].to_list())

    input_df_index += [tmp_visitor for i in range(len(tmp_luw) - 1)]
    input_df_list += tmp_luw['product_id'][:-1].to_list()
    expected_list.append(tmp_luw['product_id'][-1])

visits_3min_df_input = pd.DataFrame(data=input_df_list, index=pd.Index(input_df_index, name='visitor_id'),
                                    columns=['product_id'])
print('\n', visits_3min_df_input)
print("Nb de lignes dans luw_input + nb expected : {}".format(len(visits_3min_df_input) + len(expected_list)),
      "Nb de visites luw min visites : {}".format(len(visits_3min_df)),
      sep='\n')

visits_3min_df_input.reset_index(inplace=True)
visits_3min_df_input['nb_visit'] = np.ones(shape=(len(visits_3min_df_input), 1))
mvis_3_input = pd.pivot(visits_3min_df_input, index=["visitor_id"], columns=['product_id'], values=['nb_visit'])
mvis_3_input = mvis_3_input.fillna(0).convert_dtypes()
# print(mvis_3_input)

if mvis_3_input.sum().sum() != (len(visits_3min_df_input)):
      print("ERROR : loss of data - see code for more information")
mvis_3_input.to_csv("data/mvis_3.csv")

""" ******************************************************************************** """
""" TABLE DE CORRESPONDANCE                                                          """
""" ******************************************************************************** """

product_ids_df = pd.DataFrame(data=expected_list, columns=["product_id"])
product_ids_df = product_ids_df.drop_duplicates().reset_index(drop=True)
# print(product_ids_df)
dict_products_corresp_int_id = dict(zip(product_ids_df.index, product_ids_df['product_id']))
dict_products_corresp_id_int = dict(zip(product_ids_df['product_id'], product_ids_df.index))
# print(dict_products_corresp_int_id)
# print(dict_products_corresp_id_int)

expected_list_int = list(map(lambda x: dict_products_corresp_id_int[x], expected_list))
# print(expected_list)
# print(expected_list_int)

""" ******************************************************************************** """
""" LE MODELE                                                          """
""" ******************************************************************************** """

X = mvis_3_input.values
y = np.array(expected_list_int)
print(X, y, sep='\n')
print(len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_test.shape : {}".format(X_test.shape),
      "y_test.shape : {}".format(y_test.shape),
      "X_train.shape : {}".format(X_train.shape),
      "y_train.shape : {}".format(y_train.shape),
      sep='\n')
print(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=5)
