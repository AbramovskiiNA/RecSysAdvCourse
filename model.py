import json
from typing import Dict, List

import numpy as np
from tensorflow.keras import models


def json_to_dict(fp: str) -> Dict:
    d = json.load(open(fp))

    return {int(k): v for k, v in d.items()}


class ModelServing:
    def __init__(self):
        self.user_map = json_to_dict('user_map.json')
        self.rev_items_map = json_to_dict('rev_items_map.json')

        self.users = list(self.user_map.keys())
        self.item_encs = list(self.rev_items_map.keys())

        self.model = models.load_model('model')

    def predict(self, user_id: int, k: int = 10) -> List[int]:
        user_enc = self.user_map[user_id]

        user_item = np.array([[user_enc] * len(self.item_encs),
                              list(self.item_encs)]).T
        ratings = self.model.predict(user_item.astype('int64')).flatten()
        top_items_enc = ratings.argsort()[-k:][::-1]

        return list(map(self.rev_items_map.get, top_items_enc))


class ProductNames:
    def __init__(self):
        self.product_map = json_to_dict('products_map.json')

    def ids_to_names(self, ids: List[int]) -> List[str]:
        return list(map(self.product_map.get, ids))


if __name__ == '__main__':
    a = ModelServing()
    pred = a.predict(201052)
    print(pred)

    pn = ProductNames()
    pred_names = pn.ids_to_names(pred)
    print(pred_names)
