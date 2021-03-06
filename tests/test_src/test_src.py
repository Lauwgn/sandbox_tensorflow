import unittest
import numpy as np
import pandas as pd

from src.src import convert_vect_into_ids, correspondance_table_product_id, correspondance_table_category, \
    split_path_and_last_product, split_path_and_two_last_products, search_max_occurences


class TestConvertVectIntoIds(unittest.TestCase):

    def test_1(self):

        x = np.array([0, 0, 1, 0, 0, 0, 1])
        dict_int_id = {0: "id_0", 1: "id_1", 2: "id_2", 3: "id_3", 4: "id_4", 5: "id_5", 6: "id_6"}

        result = convert_vect_into_ids(x, dict_int_id)
        # print(result)

        self.assertEqual(["id_2", "id_6"], result)


class TestCorrespondanceTableId(unittest.TestCase):

    def test_1(self):

        id_list = ["id_1", "id_2", "id_3"]

        result1, result2 = correspondance_table_product_id(id_list)
        expected1 = {0: "id_1", 1: "id_2", 2: "id_3"}
        expected2 = {"id_1": 0, "id_2": 1, "id_3": 2}
        self.assertDictEqual(expected1, result1)
        self.assertDictEqual(expected2, result2)


class TestCorrespondanceTableCategory(unittest.TestCase):

    def test_1(self):

        cat_list = ["id_1", "id_1", "id_1", "id_2", "id_3"]

        result1, result2 = correspondance_table_category(cat_list)
        expected1 = {0: "id_1", 1: "id_2", 2: "id_3"}
        expected2 = {"id_1": 0, "id_2": 1, "id_3": 2}
        self.assertDictEqual(expected1, result1)
        self.assertDictEqual(expected2, result2)


class TestSearchMaxOccurences(unittest.TestCase):

    def test_1(self):
        l = [0, 1, 2, 3, 2]
        result = search_max_occurences(l)
        # print(result)
        self.assertEqual(2, result)

    def test_2(self):
        l = ['a', 'b', 'c', 'c']
        result = search_max_occurences(l)
        # print(result)
        self.assertEqual('c', result)


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

        # print(luw)

        visitors, visits_min_df_input, expected_list = split_path_and_last_product(luw, is_test=True)
        # print(visitors)
        self.assertEqual(["wvi_1", "wvi_2", "wvi_3"], visitors.tolist())

        # print(visits_min_df_input)
        self.assertEqual(["wvi_1", "wvi_1", "wvi_1", "wvi_2", "wvi_2", "wvi_3"], visits_min_df_input.index.tolist())
        self.assertEqual(["id_1", "id_2", "id_3", "id_2", "id_3", "id_3"], visits_min_df_input["product_id"].tolist())

        # print(expected_list)
        self.assertEqual(["id_4", "id_4", "id_4"], expected_list)


class TestSplitPathAndTwoLastProduct(unittest.TestCase):

    def test_1(self):

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3',
                                               'wvi_1', 'wvi_2', 'wvi_3'],
                                       'product_id': ['id_1',
                                               'id_2', 'id_2', 'id_5',
                                               'id_3', 'id_3', 'id_6',
                                               'id_4', 'id_4', 'id_4']
                                       })

        # print(luw)

        visitors, luw_path, prev_last_product_list, last_product_list = split_path_and_two_last_products(luw, is_test=True)

        self.assertEqual(["wvi_1", "wvi_2", "wvi_3"], visitors.tolist())

        # print(visits_min_df_input)
        self.assertEqual(["wvi_1", "wvi_1", "wvi_2", "wvi_3"], luw_path.index.tolist())
        self.assertEqual(["id_1", "id_2", "id_2", "id_5"], luw_path["product_id"].tolist())

        # print(expected_list)
        self.assertEqual(["id_3", "id_3", "id_6"], prev_last_product_list)
        self.assertEqual(["id_4", "id_4", "id_4"], last_product_list)
