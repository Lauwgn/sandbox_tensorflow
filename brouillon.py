import numpy as np
import pandas as pd

from models.catalog_manager import CatalogManager

from src.src import convert_vect_into_ids, convert_id_into_category, search_max_occurences, split_path_and_last_product


catalog = CatalogManager.import_from_json('data/20211206-catalog-533d1d6652e1-fr-en.json')

df = catalog.azimut_extract_category_to_dataframe()

df.to_csv("data/catalog_azimut_cat.csv", index=False)


# prod = CatalogManager.find_product_in_catalog_with_attributs(catalog, attribut="id", attr_value='620cd6755e67294920ba69ea')
# print(prod.to_dict())

# print("620cd6755e67294920ba69ea" in catalog.products_id_list)


# for tmp in ['6021782a360e2c9356a840ef', '60261199360e2c9356666dc7', '60218e4f360e2c9356a5f048', '602133f8360e2c93561fdc3f', '6022235d360e2c9356a8a659', '60217a9b360e2c9356c5d9d6', '620cd6755e67294920ba69ea']:
#     convert_id_into_category(tmp, catalog)