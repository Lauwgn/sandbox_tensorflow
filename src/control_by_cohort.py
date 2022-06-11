import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models.luw_manager import LuwManager
from models.mvisdense_manager import MvisDenseManager

from src.src import split_path_and_last_product


def control_by_cohort(model, dict_products_corresp_int_id, dict_products_corresp_id_int, luw, curlr):
    """ ******************************************************************************** """
    """ NE GARDER QUE LES PRODUITS "CONNUS"                                              """
    """ ******************************************************************************** """

    # product_id_list = np.intersect1d(list(dict_products_corresp_int_id.values()), curlr.index).tolist()
    product_id_list = (list(dict_products_corresp_int_id.values()))
    print("Nb products kept : {}".format(len(product_id_list)))

    luw.filter_product_ids_in_list_of_ids(product_id_list)
    # print(luw)

    """ ******************************************************************************** """
    """ VISITEURS AYANT VU AU MOINS xxx PRODUITS                                         """
    """ ******************************************************************************** """
    visits_min_df = LuwManager.select_visitors_enough_visits(luw, 3, 10)
    # print(visits_min_df)

    """ ******************************************************************************** """
    """ SPLIT : DATA X // EXPECTED RESULT y                                              """
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

    # print(y)
    y_id = np.apply_along_axis(lambda x: dict_products_corresp_int_id[x[0]], 0, [y])    # je ne comprends pas tout mais ca marche
    # print(y_id)
    y_cohorts = np.apply_along_axis(lambda x: curlr.convert_product_id_into_cohort(x[0]), 0, [y_id]) # ca marche bien, je sais pas pourquoi
    # print(y_cohorts)
    # print(pd.Series(y_cohorts).value_counts())

    predictions_id = [dict_products_corresp_int_id[np.argmax(predictions[i])] for i in range(len(predictions))]
    # print(predictions_id)
    predictions_cohorts = np.apply_along_axis(lambda x: curlr.convert_product_id_into_cohort(x[0]), 0, [predictions_id])
    # print(predictions_cohorts)

    comparison_df = pd.DataFrame.from_dict({'y': y_cohorts, 'predictions': predictions_cohorts})
    comparison_df['good'] = comparison_df['y'] == comparison_df['predictions']
    print(comparison_df)
    print(comparison_df['good'].value_counts())

    print(comparison_df[comparison_df['y'] == 'cohort_2'].value_counts())
    print(comparison_df[comparison_df['y'] == 'cohort_8'].value_counts())
    print(comparison_df[comparison_df['y'] == 'cohort_5'].value_counts())

    return 0
