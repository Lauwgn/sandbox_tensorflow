# Last Check : 07/02/2022
# Resume last Check : Maj with the Catalog object of Core_modules

import unittest

from models.catalog_manager import Catalog, CatalogManager
from models.models import Product


class TestCatalogAddProduct(unittest.TestCase):
    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2")
                                    ],
                          products_id_list=["id_1", "id_2"])
        catalog.add_product(Product(id="id_3"))

        self.assertEqual(3, len(catalog.products))
        self.assertEqual("id_3", catalog.products[2].id)

        self.assertEqual(["id_1", "id_2", "id_3"], catalog.products_id_list)


class TestCatalogRemoveProduct(unittest.TestCase):
    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2"),
                                    Product(id="id_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product(Product(id="id_3"))

        self.assertEqual(2, len(catalog.products))
        self.assertEqual("id_1", catalog.products[0].id)
        self.assertEqual("id_2", catalog.products[1].id)
        self.assertEqual(["id_1", "id_2"], catalog.products_id_list)

    def test_2_product_not_in_catalog(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2"),
                                    Product(id="id_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product(Product(id="id_4"))

        self.assertEqual(3, len(catalog.products))
        self.assertEqual(["id_1", "id_2", "id_3"], catalog.products_id_list)


class TestCatalogRemoveProductWithProductIdOnly(unittest.TestCase):
    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2"),
                                    Product(id="id_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product_with_product_id_only("id_3")

        self.assertEqual(2, len(catalog.products))
        self.assertEqual("id_1", catalog.products[0].id)
        self.assertEqual("id_2", catalog.products[1].id)
        self.assertEqual(["id_1", "id_2"], catalog.products_id_list)

    def test_2_product_not_in_catalog(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2"),
                                    Product(id="id_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product_with_product_id_only("id_4")

        self.assertEqual(3, len(catalog.products))
        self.assertEqual(["id_1", "id_2", "id_3"], catalog.products_id_list)


class TestCatalogRemoveProductWithProductUrlOnly(unittest.TestCase):
    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1", url="url_1"),
                                    Product(id="id_2", url="url_2"),
                                    Product(id="id_3", url="url_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product_with_product_url_only("url_3")

        self.assertEqual(2, len(catalog.products))
        self.assertEqual("id_1", catalog.products[0].id)
        self.assertEqual("id_2", catalog.products[1].id)
        self.assertEqual(["id_1", "id_2"], catalog.products_id_list)

    def test_2_product_not_in_catalog(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1", url="url_1"),
                                    Product(id="id_2", url="url_2"),
                                    Product(id="id_3", url="url_3")
                                    ],
                          products_id_list=["id_1", "id_2", 'id_3'])
        catalog.remove_product_with_product_url_only("url_4")

        self.assertEqual(3, len(catalog.products))
        self.assertEqual(["id_1", "id_2", "id_3"], catalog.products_id_list)


class TestCatalogToDict(unittest.TestCase):

    def test_1(self):
        tmp_catalog = Catalog(products=[Product(id="id_1")])
        result = tmp_catalog.to_dict()

        self.assertDictEqual({
            'products': [{
                        'id': "id_1",
                        'ref': None,
                        'url': None,
                        'name': None,
                        'labels': {},
                        'language': None,
                        'price': None,
                        'expired_at': None,
                        'template': None,
                        'nb_visitors': None,
                        'keywords': []
                    }],
            'products_id_list': ['id_1'],
            'updated_at': None,
            'wti': None
        }, result)


class TestCatalogGenerateIdList(unittest.TestCase):

    def test_1(self):
        catalog = Catalog(wti="",
                          products=[Product(id="id_1"),
                                    Product(id="id_2")
                                    ])
        catalog.generate_id_list()
        self.assertEqual(["id_1", "id_2"], catalog.products_id_list)

    def test_2_empty_catalog(self):
        catalog = Catalog(wti="")
        catalog.generate_id_list()
        self.assertEqual([], catalog.products_id_list)


class TestSelecIdsByDate(unittest.TestCase):

    def test_1(self):

        catalog = Catalog(products=[
                            Product(id="id_1",
                                    expired_at="2021-04-06T15:02:07.374",
                                    url="url_product_1"),
                            Product(id="id_1_bis",
                                    expired_at="2021-04-06T15:02:07.374",
                                    url="url_product_1"),
                            Product(id="id_2",
                                    expired_at="2021-04-07T15:02:07.374",
                                    url="url_product_1"),
                            Product(id="id_3",
                                    expired_at="2021-04-08T15:02:07.374",
                                    url="url_product_2"),
                            Product(id="id_4",
                                    expired_at="2021-04-09T15:02:07.374",
                                    url="url_product_2"),
                            Product(id="id_5",
                                    expired_at="2021-04-09T15:02:07.375",
                                    url="url_product_2")
                        ],
                        updated_at="2021-04-06T15:02:07.374",
                        wti="55ee9f613ece"
                    )
        catalog.generate_id_list()

        date_min = "2021-04-06T15:02:07.375"
        date_max = "2021-04-09T15:02:07.374"

        catalog.select_ids_by_dates(date_min, date_max)

        self.assertEqual(3, len(catalog.products))
        self.assertEqual(['id_2', 'id_3', 'id_4'], catalog.products_id_list)
        self.assertEqual(['id_2', 'id_3', 'id_4'], [catalog.products[i].id for i in range(3)])
