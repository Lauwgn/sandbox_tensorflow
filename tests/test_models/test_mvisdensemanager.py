import unittest
import numpy as np
import pandas as pd

from models.models import MvisDense
from models.mvisdense_manager import MvisDenseManager


class TestMakeMvis(unittest.TestCase):

    def test_1_global(self):
        luw = pd.DataFrame.from_dict({'product_id': ['id_1', 'id_3', 'id_2', 'id_1', 'id_2', 'id_3'],
                                       'visitor_id': ['wvi_1', 'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4', 'wvi_4']
                                       })

        result = MvisDenseManager.make_mvisdense(luw)
        # print(result)

        self.assertIsInstance(result, MvisDense)
        self.assertCountEqual(result.index, ['wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'])
        self.assertCountEqual(result['id_1'], [1, 0, 1, 0])
        self.assertCountEqual(result['id_2'], [0, 1, 0, 1])
        self.assertCountEqual(result['id_3'], [1, 0, 0, 1])

    def test_2_products_to_add(self):
        luw = pd.DataFrame.from_dict({'product_id': ['id_1', 'id_3', 'id_2', 'id_1', 'id_2', 'id_3'],
                                      'visitor_id': ['wvi_1', 'wvi_1', 'wvi_2', 'wvi_3', 'wvi_4', 'wvi_4']
                                      })

        result = MvisDenseManager.make_mvisdense(luw, ['id_4', 'id_5'])
        # print(result)

        self.assertIsInstance(result, MvisDense)
        self.assertCountEqual(result.index, ['wvi_1', 'wvi_2', 'wvi_3', 'wvi_4'])
        self.assertCountEqual(result['id_1'], [1, 0, 1, 0])
        self.assertCountEqual(result['id_2'], [0, 1, 0, 1])
        self.assertCountEqual(result['id_3'], [1, 0, 0, 1])
        self.assertCountEqual(result['id_4'], [0, 0, 0, 0])
        self.assertCountEqual(result['id_5'], [0, 0, 0, 0])

