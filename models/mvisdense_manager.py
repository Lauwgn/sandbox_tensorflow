import pandas as pd
import numpy as np

from models.models import MvisDense


class MvisDenseManager:

    @staticmethod

    def make_mvisdense(luw, product_id_list=None):

        luw['nb_visit'] = np.ones(shape=(len(luw), 1))
        mvis = pd.pivot_table(luw, index=["visitor_id"], columns=['product_id'], values='nb_visit',
                              fill_value=0.0)

        if product_id_list:
            products_to_add = []
            for tmp_id in product_id_list:
                if tmp_id not in mvis.columns:
                    products_to_add.append(tmp_id)

            for tmp_id in products_to_add:
                mvis[tmp_id] = np.zeros(shape=(len(mvis), 1))

            if len(np.intersect1d(mvis.columns, product_id_list)) != len(product_id_list):
                print("ERROR : loss of data - see code for more information")

        if mvis.sum().sum() != (len(luw)):
            print("ERROR : loss of data - see code for more information")

        # print(mvis)
        # print(mvis.columns)

        return MvisDense(mvis)


