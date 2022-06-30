import pandas as pd
import collections


class Catalog:
    """
    :param products : List of products
    :param products_id_list : List of the id for each product
    """
    products = []
    products_id_list = []
    updated_at = None
    wti = None

    def __init__(self, **kwargs):

        if 'products' in kwargs:
            self.products = kwargs['products']
            self.generate_id_list()
        else:
            if 'products_id_list' in kwargs:
                self.products_id_list = kwargs['products_id_list']

        if 'updated_at' in kwargs:
            self.updated_at = kwargs['updated_at']
        if 'wti' in kwargs:
            self.wti = kwargs['wti']

    def add_product(self, product):
        self.products.append(product)
        self.products_id_list.append(product.id)

    def remove_product(self, product):
        current_id = product.id
        for tmp_prod in self.products:
            if tmp_prod.id == current_id:
                self.products.remove(tmp_prod)
                break
        if current_id in self.products_id_list:
            self.products_id_list.remove(current_id)

    def remove_product_with_product_id_only(self, product_id):
        for tmp_prod in self.products:
            if tmp_prod.id == product_id:
                self.products.remove(tmp_prod)
                break
        if product_id in self.products_id_list:
            self.products_id_list.remove(product_id)

    def remove_product_with_product_url_only(self, product_url):
        product_id = ""
        for tmp_prod in self.products:
            if tmp_prod.url == product_url:
                self.products.remove(tmp_prod)
                product_id = tmp_prod.id

        if product_id in self.products_id_list:
            self.products_id_list.remove(product_id)

    def display_import_info(self):
        print("nb products in catalog: ", len(self.products_id_list))

    def to_dict(self):
        products = []

        for p in self.products:
            products.append(p.to_dict())

        return {
            'products': products,
            'products_id_list': self.products_id_list,
            'updated_at': self.updated_at,
            'wti': self.wti
        }

    def to_dataframe(self):         # @todo : à tester

        products_df_values = []
        for p in self.products:
            tmp_prod_values = [p.id, p.url, p.url, p.nb_visitors, p.keywords]
            products_df_values.append(tmp_prod_values)
        products_df = pd.DataFrame(columns=["product_id", "name", "url", "nb_visitors", "keywords"],
                                   data=products_df_values)
        products_df.set_index(keys="product_id", inplace=True)

        return products_df

    def remove_duplicates(self):
        self.generate_id_list()
        products_to_remove = [(curr_id, count) for curr_id, count in collections.Counter(self.products_id_list).items()
                              if count > 1]
        for curr_id, count in products_to_remove:
            while count > 1:
                self.remove_product_with_product_id_only(product_id=curr_id)
                count -= 1

    def generate_id_list(self):
        products_id_list = []
        for tmp_product in self.products:
            products_id_list.append(tmp_product.id)
        self.products_id_list = products_id_list

    def select_ids_by_dates(self, date_min, date_max):

        to_delete_ids_list = []

        for tmp_product in self.products:
            tmp_date = tmp_product.expired_at
            if tmp_date:
                if not date_min <= tmp_date <= date_max:
                    to_delete_ids_list.append(tmp_product.id)

        for tmp_id in to_delete_ids_list:
            self.remove_product_with_product_id_only(tmp_id)

    def select_product_by_indice(self, indice):
        """
        Input : int
            indice : indice du produit recherché
        Output : Product
            Objet de la classe Product
        """
        if not isinstance(indice, int):
            raise TypeError("Le type indice n'est pas un entier mais un %c".format(type(indice)))

        if indice > len(self.products_id_list):
            raise ValueError("Le nombre de produit est de %s, mais l'indice est %s".format(len(self.products_id_list),
                                                                                           indice))
        return self.products[indice]

    def azimut_extract_category_to_dataframe(self):

        id_list, url_list, ref_list, category_list = [], [], [], []

        for tmp_prod in self.products:
            tmp_id = tmp_prod.id
            tmp_url = tmp_prod.url
            tmp_ref = tmp_prod.ref
            tmp_cat = tmp_prod.convert_into_category_azimut()

            id_list.append(tmp_id)
            url_list.append(tmp_url)
            ref_list.append(tmp_ref)
            category_list.append(tmp_cat)

        df = pd.DataFrame.from_dict(
            {"product_id": id_list, "url": url_list, "category": category_list, "ref": ref_list})
        print(df['category'].isnull().value_counts())

        df.sort_values(by='category', ascending=True, inplace=True)
        df.to_csv("data/catalog_azimut_cat.csv", index=False)

        return df




    @staticmethod
    def strip_url(x):
        """
           Critère de coupe d'une url
           Input : string or float

           Return: string
           String coupé à un certain endroit
           """

        # Si ce n'est pas une suite de caractères, souvent nan, on renvoie rien
        if type(x) is not str:
            return ""

        if x.find("https://www.normandie-tourisme.fr") != -1:
            # if x.find("%") != -1:
            #     x = x.split("%")[0]
            if x.find("#") != -1:
                x = x.split("#")[0]
            if x.find("?") != -1:
                x = x.split("?")[0]
            # if x.find("&") != -1:
            #     x = x.split("&")[0]

            # Si la fin de l'url est un caractère spécial
            if not x[-1].isalnum():
                if x[-2:] == "//":
                    x = x[:-1]
                if x[-1] != "/":
                    x = x[:-1]
            return x

        else:
            return ""
