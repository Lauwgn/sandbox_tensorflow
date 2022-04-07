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


    @staticmethod
    def bad_url_crtnormandie(x):
        """
           Critère d'une url
           Input : string or float
           critères : list
            Liste de caractères prohibées une url

           Return: boolean
               Si True, l'url est mauvaise, sinon on garde l'url lié à un produit
           """

        # Si ce n'est pas une suite de caractères, souvent nan, on renvoie rien
        if type(x) is not str:
            return True

        if x.find("https://www.normandie-tourisme.fr") != -1:
            if x.find("#") != -1:
                return True
            if x.find("?") != -1:
                return True

            # Si la fin de l'url est un caractère spécial
            if not x[-1].isalnum():
                if x[-2:] == "//":
                    return True
                if x[-1] != "/":
                    return True
            return False

        else:
            return True

    @staticmethod
    def bad_url(x, to_remove=[], to_keep=[]):
        """ Verifie si l'url d'entrée convient

        Parameters
        ----------
        x : string or float
            String or float, but should be a url
        to_remove : list
            Liste de caractères prohibés une url
        to_keep : list
            Liste des parties d'urls obligatoires, par exemple "https://www.legrandbornand.com/"
            est une partie obligatoire pour GB OT
            Si keep est vide, on ne considère pas de filtre là dessus
        Returns
        -------
        cohort_wvi_list : List of string
            Contains all the wvi in the cohort
        """

        # Si ce n'est pas une suite de caractères, souvent nan, mauvaise addresse
        if type(x) is not str:
            return True

        if to_keep:     # Liste non vide
            to_keep_boolean = [str(x.find(url_keep) != -1) for url_keep in to_keep]

            try:    # Si ça ne plante pas, alors il y a effectivement un True dans la liste to_keep_boolean
                result = to_keep_boolean.index("True") != -1
            except ValueError:          # Si on a aucun éléments de to_keep qui correspond à x
                                        # alors on retourne True car il ne trouvera pas la valeur True dans la liste
                return True

        for remove in to_remove:
            if x.find(remove) != -1:  # Si le caractère est présent

                return True

        return False


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
