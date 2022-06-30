import numpy as np
import pandas as pd
from collections import Counter

""" DETERMINE MAIN COHORT FROM A PATH OF A VISITOR (WVI or VISITOR_ID) """


def determine_main_category_luw_sorted(wvi_path, dict_id_to_cohort=None, include_minus_1=False):
    """Retrieve the main cohort of a wvi by finding the last id seen in a candidate cohort

    Parameters
    ----------
    wvi_path : list of string or int
        List of ids in a temporal sequence by a visitor if the luw is sorted
    dict_id_to_cohort : dict
        Dictionnaire ayant un product_id (key) lié à une cohorte (value)
    include_minus_1 : bool
        Choix de l'inclusion ou non de la cohorte -1 dans le choix de la cohorte principale d'un visiteur
    Returns
    -------
    string
        The main cohort of a wvi

    """

    wvi_main_category = None

    list_ids = list(dict_id_to_cohort)
    list_wvi_conversion_cohort = [dict_id_to_cohort[curr_id] for curr_id in wvi_path if curr_id in list_ids]
    list_wvi_cohort = Counter(list_wvi_conversion_cohort)

    if len(list_wvi_cohort) != 0:
        list_wvi_main_category = retrieve_most_occured_category(list_wvi_cohort)
        if len(list_wvi_main_category) >= 2:        # Même nombre de produits vus dans une cohorte qu'une autre
            try:
                wvi_main_category = retrieve_most_recent_category(list_wvi_conversion_cohort, list_wvi_main_category)

            except ValueError as e:
                print(f"Error ici : {e}")
                print(f"list_wvi_conversion_cohort : {list_wvi_conversion_cohort}")
                print(f"list_wvi_main_cohort : {list_wvi_main_category}")

        else:
            wvi_main_category = list_wvi_main_category[0]

    return wvi_main_category          # <class string>


def retrieve_most_recent_category(list_wvi_conversion_cohort, list_wvi_main_category):
    """Retrieve the main cohort of a wvi by finding the last id seen in a candidate cohort

    Parameters
    ----------
    list_wvi_conversion_cohort : :list of string
        The temporal order of the id seen by a wvi, transformed to a list of cohort
    list_wvi_main_category : list of string
        Represente the candidates for the main_cohort

    Returns
    -------
    string
        The main cohort of a wvi

    """
    list_copy = list_wvi_conversion_cohort.copy()       # On evite que notre liste originale soit inversée
    list_copy.reverse()     # Les cohortes visitées les plus récentes sont en premier dans la liste

    # Indice de la première occurence pour chaque cohorte
    recency_index = [list_copy.index(main_cohort) for main_cohort in list_wvi_main_category]
    # On recupère la cohorte qui apparaît le premier dans la liste
    most_recent_cohort = np.argmin(recency_index)
    wvi_main_category = list_wvi_main_category[most_recent_cohort]

    return wvi_main_category


def retrieve_most_occured_category(list_wvi_cohort):
    """Retrieve all candidate cohort which appears the most in a Counter type

    Parameters
    ----------
    list_wvi_cohort : :Dict from Counter collections
        Dict of each cohort associated with the number of occurences : (str, int)

    Returns
    -------
    List of string
        The candidate(s) to be the main cohort for a visitor

    """
    list_wvi_main_category = []

    if len(list_wvi_cohort) == 1:
        wvi_main_cohort = list_wvi_cohort.most_common(1)[0][0]
        list_wvi_main_category.append(wvi_main_cohort)

    else:
        # On vérifie s'il y a un autre maximum, on récupere toutes les cohortes associées
        i = 1
        curr_counter = list_wvi_cohort.most_common(i)[-1]   # L'occurence la plus fréquente choisie ici
        # print(list_wvi_cohort.most_common(1)[0])
        max_value = curr_counter[1]     # La valeur maximale d'occurence
        list_wvi_main_category.append(curr_counter[0])

        for i in range(2, len(list_wvi_cohort) + 1):
            curr_counter = list_wvi_cohort.most_common(i)[-1]
            if max_value == curr_counter[1]:  # S'il y a le même nombre d'occurences
                list_wvi_main_category.append(curr_counter[0])
            else:
                break

    return list_wvi_main_category


