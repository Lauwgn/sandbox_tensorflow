import unittest
import pandas as pd

from models.models import Curlr, Luw, MvisThresholds, Product
from models.model_catalog import Catalog


class TestCurlrUpdateNbVisitorsIdFromLuw(unittest.TestCase):

    def test_1_general(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_2', 'wvi_2',
                                               'wvi_3', 'wvi_3', 'wvi_3',
                                               'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_1', 'id_2',
                                               'id_1', 'id_3', 'id_4',
                                               'id_1', 'id_2', 'id_3', 'id_4']
                                       })
        luw = Luw(luw)
        # print(luw)

        curlr.update_nb_visitors_id_from_luw(luw)
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual([4, 2, 2, 2], curlr['nb_visitors_id'].tolist())

    def test_2_id_luw_not_in_curlr(self):
        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                                     'wvi_2', 'wvi_2',
                                                     'wvi_3', 'wvi_3', 'wvi_3',
                                                     'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'],
                                      'product_id': ['id_1',
                                                     'id_1', 'id_2',
                                                     'id_1', 'id_3', 'id_4',
                                                     'id_1', 'id_2', 'id_3', 'id_5']
                                      })
        luw = Luw(luw)
        # print(luw)

        curlr.update_nb_visitors_id_from_luw(luw)
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual([4, 2, 2, 1], curlr['nb_visitors_id'].tolist())
        # id luw not in curlr, et réciproquement

    def test_3_id_curlr_not_in_luw(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 5, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        luw = pd.DataFrame.from_dict({'visitor_id': ['wvi_1',
                                               'wvi_2', 'wvi_2',
                                               'wvi_3', 'wvi_3', 'wvi_3',
                                               'wvi_4', 'wvi_4', 'wvi_4', 'wvi_4'],
                                       'product_id': ['id_1',
                                               'id_1', 'id_2',
                                               'id_1', 'id_3', 'id_4',
                                               'id_1', 'id_2', 'id_3', 'id_4']
                                       })
        luw = Luw(luw)
        # print(luw)

        curlr.update_nb_visitors_id_from_luw(luw)
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual([4, 2, 2, 2, 5], curlr['nb_visitors_id'].tolist())


class TestCurlrUpdateCohortSize(unittest.TestCase):

    def test_1(self):

        columns = ['cohort_id', 'qualif', 'cohort_size']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4],
                ["cohort_-1", "ISOLES", 4],
                ["cohort_0", "qualif_0", 3],
                ["cohort_1", "qualif_1", 2]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.update_cohort_size()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual([2, 2, 1, 1], curlr['cohort_size'].tolist())

    def test_2_curlr_empty(self):

        curlr = Curlr(pd.DataFrame())
        # print(curlr)

        curlr.update_cohort_size()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(0, len(curlr))


class TestCurlrSortForExportCohortSizeAndNbVisitorsId(unittest.TestCase):

    def test_1(self):

        columns = ['cohort_id', 'qualif', 'cohort_size', 'nb_visitors_id', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 3, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 1, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 1, 3, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 1, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 3, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.sort_for_export_cohort_size_and_nb_visitors_id()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(["id_1", "id_5", "id_4", "id_2", "id_3"], curlr.index.tolist())


class TestCurlrSortForExportFull(unittest.TestCase):

    def test_1_nb_visitors_coh_and_nb_visitors_id(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 4, 12, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 5, 12, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 8, 1, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 10, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.sort_for_export_full()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(["id_3", "id_2", "id_1", "id_5", "id_4"], curlr.index.tolist())

    def test_2_nb_visitors_coh(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 10, 1, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 12, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 12, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 8, 1, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 10, 3, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.sort_for_export_full()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(["id_2", "id_3", "id_5", "id_1", "id_4"], curlr.index.tolist())

    def test_3_nb_visitors_id(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 4, 12, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 5, 12, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 8, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 10, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.sort_for_export_full()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(["id_3", "id_2", "id_1", "id_5", "id_4"], curlr.index.tolist())


    def test_4_no_nb_visitors_coh_and_no_nb_visitors_id(self):

        columns = ['cohort_id', 'qualif', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4", "id_5"], name="id")
        data = [["cohort_-1", "ISOLES", 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 1, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 1, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))
        # print(curlr)

        curlr.sort_for_export_full()
        # print(curlr)

        self.assertIsInstance(curlr, Curlr)
        self.assertEqual(["id_1", "id_4", "id_5", "id_2", "id_3"], curlr.index.tolist())


class TestLuwFilterProductIdsFromCatalog(unittest.TestCase):

    def test_1_general(self):

        catalog = Catalog(products_id_list=["id_1", "id_3"])

        luw = Luw(data=[["wvi_1", "id_1"],
                        ["wvi_2", "id_2"]],
                  columns=["visitor_id", "product_id"])

        luw.filter_product_ids_from_catalog(catalog)

        self.assertEqual(1, len(luw))
        self.assertEqual("id_1", luw["product_id"].iloc[0])

    def test_2_empty_luw(self):

        catalog = Catalog(products_id_list=["id_1", "id_3"])

        luw = Luw(columns=["visitor_id", "product_id"])

        luw.filter_product_ids_from_catalog(catalog)

        self.assertEqual(0, len(luw))

    def test_3_empty_catalog(self):

        catalog = Catalog()

        luw = Luw(data=[["wvi_1", "id_1"],
                        ["wvi_2", "id_2"]],
                  columns=["visitor_id", "product_id"])

        luw.filter_product_ids_from_catalog(catalog)

        self.assertEqual(0, len(luw))

    def test_4_all_products_in_catalog(self):
        catalog = Catalog(products_id_list=["id_1", "id_2"])

        luw = Luw(data=[["wvi_1", "id_1"],
                        ["wvi_2", "id_2"]],
                  columns=["visitor_id", "product_id"])

        luw.filter_product_ids_from_catalog(catalog)

        self.assertEqual(2, len(luw))
        self.assertEqual("id_1", luw["product_id"].iloc[0])
        self.assertEqual("id_2", luw["product_id"].iloc[1])


class TestMvisThresholdsInit(unittest.TestCase):

    def test_init(self):

        x = MvisThresholds()
        # print(x)
        #
        # x = MvisThresholds(nb_visits_min_threshold_id=10)
        # print(x.nb_visits_min_threshold_id)
        # print(x.nb_visits_max_threshold_id)
        #
        # x = MvisThresholds(nb_visits_min=10)
        # print(x.nb_visits_min_threshold_id)
        # print(x.nb_visits_max_threshold_id)
        #
        # x = MvisThresholds(10)
        # print(x.nb_visits_min_threshold_id)
        # print(x.nb_visits_max_threshold_id)


class TestMvisThresholdsToDict(unittest.TestCase):

    def test_1(self):

        x = MvisThresholds(nb_visits_min_threshold_id=5, nb_visits_min_threshold_visitors=10)
        self.assertDictEqual(x.to_dict(),
                             {'nb_visits_min_threshold_id': 5,
                                'nb_visits_max_threshold_id': 0,
                                'nb_visits_min_threshold_visitors': 10,
                                'nb_visits_max_threshold_visitors': 0})


class TestProductToDict(unittest.TestCase):

    def test_1(self):
        tmp_product = Product(id=123, url="url_1")
        result = tmp_product.to_dict()

        self.assertEqual({
            'id': 123,
            'ref': None,
            'url': "url_1",
            'name': None,
            'labels': {},
            'language': None,
            'price': None,
            'expired_at': None,
            'template': None,
            'nb_visitors': None,
            'keywords': [],
        }, result)

