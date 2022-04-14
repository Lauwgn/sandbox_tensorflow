# Last Check : 17/01/2021
# Resume last Check : Check Test

import unittest

import pandas as pd

from models.models import Luw, Mvis
from models.luw_manager import LuwManager


class TestLuwSelectProductIdsEnoughVisits(unittest.TestCase):

    def test_1_not_too_many_id_and_enough_id(self):
        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_1', 'wvi_2',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_2', 'id_2',
                                               'id_3', 'id_3', 'id_3',
                                               'id_4', 'id_4', 'id_4', 'id_4']
                                       })
        luw = Luw(luw)

        nb_visits_min_threshold_id = 2

        nb_visits_max_threshold_id = 3
        result = LuwManager.select_product_ids_enough_visits(luw, nb_visits_min_threshold_id, nb_visits_max_threshold_id, is_test=True)
        # print(result)

        self.assertEqual(len(result), 5)
        self.assertCountEqual(result['product_id'], ['id_2', 'id_2', 'id_3', 'id_3', 'id_3'])

    def test_2_threshold_none(self):
        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_1', 'wvi_2',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_2', 'id_2',
                                               'id_3', 'id_3', 'id_3',
                                               'id_4', 'id_4', 'id_4', 'id_4']
                                       })
        luw = Luw(luw)

        nb_visits_min_threshold_id = 2

        nb_visits_max_threshold_id = None
        result = LuwManager.select_product_ids_enough_visits(luw, nb_visits_min_threshold_id, nb_visits_max_threshold_id, is_test=True)
        # print(result)

        self.assertEqual(len(result), 9)
        self.assertCountEqual(result['product_id'], ['id_2', 'id_2', 'id_3', 'id_3', 'id_3', 'id_4', 'id_4', 'id_4', 'id_4'])

    def test_3_luw_data_empty(self):
        luw = pd.DataFrame.from_dict({'visitor_id': [],
                                       'product_id': []
                                       })
        luw = Luw(luw)

        nb_visits_min_threshold_id = 2

        nb_visits_max_threshold_id = 3
        result = LuwManager.select_product_ids_enough_visits(luw, nb_visits_min_threshold_id, nb_visits_max_threshold_id, is_test=True)
        # print(result)

        self.assertEqual(len(result), 0)


class TestLuwSelectVisitorIdsEnoughVisits(unittest.TestCase):

    def test_1_not_too_many_wvi_and_enough_id(self):
        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_2', 'wvi_2',
                                               'wvi_3', 'wvi_3', 'wvi_3',
                                               'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_1', 'id_1',
                                               'id_1', 'id_1', 'id_1',
                                               'id_1', 'id_1', 'id_1', 'id_1']
                                       })
        luw = Luw(luw)
        nb_visits_min_threshold_visitors = 2
        nb_visits_max_threshold_visitors = 3
        result = LuwManager.select_visitors_enough_visits(luw, nb_visits_min_threshold_visitors,
                                                          nb_visits_max_threshold_visitors, is_test=True)
        # print(result)
        self.assertEqual(len(result), 5)
        self.assertCountEqual(result['visitor_id'], ['wvi_2', 'wvi_2', 'wvi_3', 'wvi_3', 'wvi_3'])

    def test_2_threshold_none(self):
        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_2', 'wvi_2',
                                               'wvi_3', 'wvi_3', 'wvi_3',
                                               'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_1', 'id_1',
                                               'id_1', 'id_1', 'id_1',
                                               'id_1', 'id_1', 'id_1', 'id_1']
                                       })
        luw = Luw(luw)
        nb_visits_min_threshold_visitors = 2
        nb_visits_max_threshold_visitors = None
        result = LuwManager.select_visitors_enough_visits(luw, nb_visits_min_threshold_visitors,
                                                          nb_visits_max_threshold_visitors, is_test=True)
        # print(result)
        self.assertEqual(len(result), 9)
        self.assertCountEqual(result['visitor_id'], ['wvi_2', 'wvi_2', 'wvi_3', 'wvi_3', 'wvi_3',
                                                     'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'])

    def test_3_luw_data_empty(self):
        luw = pd.DataFrame.from_dict({'visitor_id': [],
                                       'product_id': []
                                       })
        luw = Luw(luw)
        nb_visits_min_threshold_visitors = 2
        nb_visits_max_threshold_visitors = 3
        result = LuwManager.select_visitors_enough_visits(luw, nb_visits_min_threshold_visitors,
                                                          nb_visits_max_threshold_visitors, is_test=True)
        # print(result)
        self.assertEqual(len(result), 0)


class TestConvertLuwIntoMvisSparse(unittest.TestCase):

    def test_1(self):

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1', 'wvi_2', 'wvi_2', 'wvi_3'],
                                       'product_id': ['id_1', 'id_1', 'id_2', 'id_1']
                                       })
        luw = Luw(luw)

        result = LuwManager.convert_luw_into_mvis_sparse(luw)

        # print(result)

        self.assertIsInstance(result, Mvis)
        self.assertEqual(['wvi_1', 'wvi_2', 'wvi_3'], result.visitor_ids_list)
        self.assertEqual(['id_1', 'id_2'], result.product_ids_list)
        values = result.values
        # print(values.todense())
        self.assertEqual([[1, 0], [1, 1], [1, 0]], values.todense().tolist())

    def test_2_luw_empty(self):

        luw = pd.DataFrame.from_dict({'visitor_id': [],
                                       'product_id': []
                                       })
        luw = Luw(luw)

        result = LuwManager.convert_luw_into_mvis_sparse(luw)

        # print(result)

        self.assertIsInstance(result, Mvis)
        self.assertEqual([], result.visitor_ids_list)
        self.assertEqual([], result.product_ids_list)
        values = result.values
        self.assertIsNone(values)


class TestFilterListWithOnlyIdsInLuw(unittest.TestCase):

    def test_1(self):

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1', 'wvi_2', 'wvi_2', 'wvi_3'],
                                       'product_id': ['id_1', 'id_1', 'id_2', 'id_1']
                                       })
        luw = Luw(luw)

        id_list = ["id_1", "id_3"]

        result = LuwManager.filter_list_with_only_ids_in_luw(luw, id_list)

        self.assertEqual(["id_1"], result)