import unittest
import pandas as pd
import numpy as np

from models.models import Curlr, Luw, MvisDense, MvisThresholds, Product
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
        # id luw not in curlr, et r??ciproquement

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



class TestCurlrConvertProductIdIntoCohort(unittest.TestCase):

    def test_1(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))

        product_id = "id_1"

        result = curlr.convert_product_id_into_cohort(product_id)
        # print(result)

        self.assertEqual("cohort_-1", result)

    def test_2_none(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))

        product_id = "id_5"

        result = curlr.convert_product_id_into_cohort(product_id)
        # print(result)

        self.assertIsNone(result)



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


class TestLuwAddColumnCategory(unittest.TestCase):

    def test_1(self):
        luw = Luw(data=[["wvi_1", "id_1"],
                        ["wvi_2", "id_5"],
                        ["wvi_3", "id_3"]],
                  columns=["visitor_id", "product_id"])

        catalog_df = pd.DataFrame([["id_1", 'cat_1'], ["id_2", 'cat_2'], ["id_3", 'cat_3'], ["id_4", 'cat_4']],
                                  columns=["product_id", "category"])
        catalog_df = catalog_df.set_index(keys='product_id')
        # print(catalog_df)

        luw.add_column_category(catalog_df)
        # print(luw)
        result = luw['category'].tolist()
        self.assertEqual(['cat_1', None, 'cat_3'], result)


class TestMvisDenseRenameColumnsToInt(unittest.TestCase):

    def test_1(self):

        visitors_list = ["wvi_1", "wvi_2", "wvi_3"]
        id_list = ["id_1", "id_3", "id_2"]
        values = np.array([[0, 0, 0],
                           [1, 1, 0],
                           [1, 0, 0]])
        mvis = MvisDense(pd.DataFrame(index=visitors_list,
                                      columns=id_list,
                                      data=values))
        # print(mvis)

        dict_products_corresp_id_int = {"id_1": 0, "id_2": 1, "id_3": 2}

        mvis.rename_columns_to_int(dict_products_corresp_id_int)
        # print(mvis)

        self.assertIsInstance(mvis, MvisDense)
        self.assertEqual(["wvi_1", "wvi_2", "wvi_3"], mvis.index.tolist())
        self.assertEqual([0, 1, 2], mvis.columns.tolist())
        self.assertEqual([0, 0, 0, 1, 0, 1, 1, 0, 0], np.reshape(mvis.values, (1, -1))[0].tolist())


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


class TestProductConvertIntCategoryAzimut(unittest.TestCase):

    def test_1(self):

        p = Product(id="id_1", ref="FRSR01")
        # print(p.to_dict())

        result = p.convert_into_category_azimut()
        # print(result)

        self.assertEqual("SR", result)


class TestProductConvertIntoCohort(unittest.TestCase):

    def test_1(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))

        p = Product(id="id_1", ref="FRSR01")

        result = p.convert_into_cohort(curlr)
        # print(result)

        self.assertEqual("cohort_-1", result)

    def test_2_none(self):

        columns = ['cohort_id', 'qualif', 'nb_visitors_id', 'nb_visitors_coh', 'cohort_size', 'url', 'name']
        index = pd.Index(data=["id_1", "id_2", "id_3", "id_4"], name="id")
        data = [["cohort_-1", "ISOLES", 4, 10, 3, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_0", "qualif_0", 3, 8, 2, "url_1", "name_1"],
                ["cohort_1", "qualif_1", 2, 6, 1, "url_1", "name_1"]]

        curlr = Curlr(pd.DataFrame(data=data, columns=columns, index=index))

        p = Product(id="id_5", ref="FRSR01")

        result = p.convert_into_cohort(curlr)
        # print(result)

        self.assertIsNone(result)


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

