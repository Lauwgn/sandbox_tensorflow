import unittest
import numpy as np
import pandas as pd

from src.src import convert_vect_into_ids, correspondance_table, make_mvis, select_visitors_enough_visits,\
    split_path_and_last_product


class TestConvertVectIntoIds(unittest.TestCase):

    def test_1(self):

        x = np.array([0, 0, 1, 0, 0, 0, 1])
        dict_int_id = {0: "id_0", 1: "id_1", 2: "id_2", 3: "id_3", 4: "id_4", 5: "id_5", 6: "id_6"}

        result = convert_vect_into_ids(x, dict_int_id)
        # print(result)

        self.assertEqual(["id_2", "id_6"], result)


class TestCorrespondanceTable(unittest.TestCase):

    def test_1(self):

        id_list = ["id_1", "id_2", "id_3"]

        result1, result2 = correspondance_table(id_list)
        expected1 = {0: "id_1", 1: "id_2", 2: "id_3"}
        expected2 = {"id_1": 0, "id_2": 1, "id_3": 2}
        self.assertDictEqual(expected1, result1)
        self.assertDictEqual(expected2, result2)


class TestMakeMvis(unittest.TestCase):

    def test_1_global(self):
        luw = pd.DataFrame.from_dict({'product_id': ['id_1', 'id_3', 'id_2', 'id_1', 'id_2', 'id_3'],
                                       'visitor_id': ['wvi_1', 'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4', 'wvi_4']
                                       })

        result = make_mvis(luw)
        # print(result)

        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertCountEqual(result.index, ['wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'])
        self.assertCountEqual(result['id_1'], [1, 0, 1, 0])
        self.assertCountEqual(result['id_2'], [0, 1, 0, 1])
        self.assertCountEqual(result['id_3'], [1, 0, 0, 1])


class TestSelectVisitorsEnoughVisits(unittest.TestCase):

    def test_1(self):

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_1', 'wvi_2',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_2', 'id_2',
                                               'id_3', 'id_3', 'id_3',
                                               'id_4', 'id_4', 'id_4', 'id_4']
                                       })

        result = select_visitors_enough_visits(luw, min_visits=2, max_visits=3)
        # print(result)

        self.assertEqual(5, len(result))
        self.assertEqual(3, len(result.loc["wvi_2"]))
        self.assertEqual(2, len(result.loc["wvi_3"]))
        self.assertEqual(["product_id"], result.columns)
        self.assertEqual("visitor_id", result.index.name)


class TestSplitPathAndLastProduct(unittest.TestCase):

    def test_1(self):

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_1', 'wvi_2',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3'],
                                       'product_id': ['id_1',
                                               'id_2', 'id_2',
                                               'id_3', 'id_3', 'id_3',
                                               'id_4', 'id_4', 'id_4']
                                       })

        luw.set_index("visitor_id", inplace=True, drop=True)
        # print(luw)

        visitors, visits_min_df_input, expected_list = split_path_and_last_product(luw)
        # print(visitors)
        self.assertEqual(["wvi_1", "wvi_2", "wvi_3"], visitors.tolist())

        # print(visits_min_df_input)
        self.assertEqual(["wvi_1", "wvi_1", "wvi_1", "wvi_2", "wvi_2", "wvi_3"], visits_min_df_input.index.tolist())
        self.assertEqual(["id_1", "id_2", "id_3", "id_2", "id_3", "id_3"], visits_min_df_input["product_id"].tolist())

        # print(expected_list)
        self.assertEqual(["id_4", "id_4", "id_4"], expected_list)

