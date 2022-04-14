import unittest
import datetime
# from pathlib import Path

from models.models import Product
from models.model_catalog import Catalog
from models.catalog_manager import CatalogManager


# class TestCatalogManagerImportFromJson(unittest.TestCase):
#
#     def test_1(self):
#         filename = str(Path(__file__).parent.parent.parent) + '/tests/test_models/data_test/catalog_test_catalogmanager_import_json.json'
#
#         catalog = CatalogManager.import_from_json(filename)
#         result = catalog.to_dict()
#
#         self.assertIsInstance(catalog, Catalog)
#         self.assertEqual(4, len(result.keys()))
#         self.assertCountEqual(["products", "products_id_list", "updated_at", "wti"], list(result.keys()))
#
#         tmp_prod = catalog.products[1]
#         self.assertIsInstance(tmp_prod, Product)
#         self.assertEqual("60213238360e2c93560e5a9f", tmp_prod.id)
#         self.assertDictEqual({
#                 "recommendable": True
#             }, tmp_prod.labels)


class TestsCatalogManagerIsProductInCatalog(unittest.TestCase):

    def test_1_true(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012"), Product(id="1013")])
        product = Product(id="1012")

        cat_manager = CatalogManager()
        result = cat_manager.is_product_in_catalog(product, catalog)

        self.assertEqual(True, result)

    def test_2_false(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012"), Product(id="1013")])
        product = Product(id="1025")

        cat_manager = CatalogManager()
        result = cat_manager.is_product_in_catalog(product, catalog)

        self.assertFalse(False, result)


class TestsCatalogManagerIsIdInCatalog(unittest.TestCase):

    def test_1_true(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012"), Product(id="1013")])
        current_id = "1012"

        cat_manager = CatalogManager()
        result = cat_manager.is_id_in_catalog(current_id, catalog)

        self.assertEqual(True, result)

    def test_2_false(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012"), Product(id="1013")])
        current_id = "1025"

        cat_manager = CatalogManager()
        result = cat_manager.is_id_in_catalog(current_id, catalog)

        self.assertFalse(False, result)


class TestCatalogManagerFilterListWithOnlyIdsInCatalog(unittest.TestCase):

    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1", url="url_1", name='name_1'),
                                    Product(id="id_2", url="url_2", name='name_2'),
                                    Product(id="id_3", url="url_3", name='name_3'),
                                    Product(id="id_4", url="url_4", name='name_4')])
        catalog.generate_id_list()
        # print(catalog.to_dict())

        id_list = ["id_1", "id_2", "id_5"]

        result = CatalogManager.filter_list_with_only_ids_in_catalog(id_list, catalog)
        # print(result)

        self.assertEqual(["id_1", "id_2"], result)

    def test_2_empty_list(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1", url="url_1", name='name_1'),
                                    Product(id="id_2", url="url_2", name='name_2'),
                                    Product(id="id_3", url="url_3", name='name_3'),
                                    Product(id="id_4", url="url_4", name='name_4')])
        catalog.generate_id_list()
        # print(catalog.to_dict())

        id_list = []

        result = CatalogManager.filter_list_with_only_ids_in_catalog(id_list, catalog)
        # print(result)

        self.assertEqual([], result)

    def test_3_catalog_empty(self):
        catalog = Catalog(wti="",
                          products=[])
        catalog.generate_id_list()
        # print(catalog.to_dict())

        id_list = ["id_1", "id_2", "id_5"]

        result = CatalogManager.filter_list_with_only_ids_in_catalog(id_list, catalog)
        # print(result)

        self.assertEqual([], result)


class TestCatalogManagerFindProductInCatalogWithId(unittest.TestCase):

    def test_1_true(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'), Product(id="1013")])
        current_id = "1012"

        result = CatalogManager.find_product_in_catalog_with_id(current_id, catalog)

        self.assertIsInstance(result, Product)
        self.assertEqual("1012", result.id)
        self.assertEqual("name_1", result.name)
        self.assertEqual(None, result.expired_at)

    def test_2_false(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'), Product(id="1013")])
        current_id = "1014"

        result = CatalogManager.find_product_in_catalog_with_id(current_id, catalog)

        self.assertIsNone(result)

    def test_3_catalog_empty(self):

        catalog = Catalog()
        current_id = "1013"

        result = CatalogManager.find_product_in_catalog_with_id(current_id, catalog)

        self.assertIsNone(result)


class TestRemoveProduct(unittest.TestCase):

    def test_1_two_products(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'),
                                    Product(id="1013", url="url_2", name='name_2')]
                          )

        current_product = catalog.select_product_by_indice(0)
        catalog.remove_product(current_product)

        self.assertEqual(1, len(catalog.products_id_list))
        self.assertEqual("1013", catalog.select_product_by_indice(0).id)
        self.assertEqual("url_2", catalog.select_product_by_indice(0).url)
        self.assertEqual("name_2", catalog.select_product_by_indice(0).name)


class TestBadUrl(unittest.TestCase):

    def test_check_bad_url(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'),
                                    Product(id="1013", url="url_2", name='name_2')]
                          )
        result = []
        for tmp_prod in catalog.products:
            result.append(catalog.bad_url(x=tmp_prod.url, to_remove=['1'], to_keep=['url']))

        # print(result)
        self.assertEqual(2, len(catalog.products_id_list))
        self.assertEqual(True, result[0])
        self.assertEqual(False, result[1])

    def test_without_to_keep(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'),
                                    Product(id="1013", url="url_2", name='name_2')]
                          )
        result = []
        for tmp_prod in catalog.products:
            result.append(catalog.bad_url(x=tmp_prod.url, to_remove=['1']))

        # print(result)
        self.assertEqual(2, len(catalog.products_id_list))
        self.assertEqual(True, result[0])
        self.assertEqual(False, result[1])

    def test_several_to_keep(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1", name='name_1'),
                                    Product(id="1013", url="url_2", name='name_2')]
                          )
        result = []
        for tmp_prod in catalog.products:
            result.append(catalog.bad_url(x=tmp_prod.url, to_remove=['3'], to_keep=['1', '2']))

        # print(result)
        self.assertEqual(2, len(catalog.products_id_list))
        self.assertEqual(False, result[0])
        self.assertEqual(False, result[1])

    def test_alpha_numerical(self):

        catalog = Catalog(wti="", updated_at=datetime.datetime.strptime("2020-12-30T16:31:45.104000", "%Y-%m-%dT%H:%M:%S.%f"),
                          products=[Product(id="1012", url="url_1/", name='name_1'),
                                    Product(id="1013", url="url_2", name='name_2'),
                                    Product(id="1014", url="url_3/new_product_asha#", name='name_3')]
                          )
        result = []
        for tmp_prod in catalog.products:
            result.append(catalog.bad_url(x=tmp_prod.url, to_remove=['#'], to_keep=['url']))

        # print(result)
        self.assertEqual(3, len(catalog.products_id_list))
        self.assertEqual(False, result[0])
        self.assertEqual(False, result[1])
        self.assertEqual(True, result[2])

