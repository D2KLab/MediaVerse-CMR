import json
from typing import Dict, List, Union

import spacy


class Filter:
    def __init__(self, concepts_table: Dict[str, List[str]]):
        self._table = concepts_table
        self._nlp   = spacy.load("en_core_web_sm")

    def query_expansion(self, query: str) -> List[str]:
        related_tags = set()
        for doc in self._nlp(query):
            lemma = doc.lemma_
            if lemma in self._table:
                #import pdb; pdb.set_trace()
                for tag in self._table[lemma]:
                    related_tags.add(tag)
        return set(related_tags)

    def filter(self, query: Union[str, Dict], pool: List):
        """
        pool is a list of dictionaries {'id': torch.Tensor, 'tags': List[str]}
        """
        if type(query) == str:
            # query is textual and pool is visual
            return self._filter_visual_pool(query, pool)
        # query is visual and pool is textual
        return self._filter_txt_pool(query, pool)

    def _filter_visual_pool(self, query: str, pool: List):
        related_tags = self.query_expansion(query)
        retrieved    = []
        for img in pool:
            if len(set(img['tags']).intersection(related_tags)) > 0:
                retrieved.append(img)
        if not len(retrieved):
            return pool
        return retrieved

    def _filter_txt_pool(self, query: Dict, pool: List[Dict]):
        """
        each element in the pool is {'id': int, 'txt': str}
        """
        retrieved = []
        for txt in pool:
            related_tags = self.query_expansion(txt['txt'])
            if len(set(query['tags']).intersection(related_tags)) > 0:
                retrieved.append(txt)
        if not len(retrieved):
            return pool
        return retrieved


if __name__ == '__main__':
    with open("cache/concepts_table.json") as fp:
        table = json.load(fp)
    filter = Filter(table)
    query = 'Mom and daughter are using the hoven'
    tags = filter.query_expansion(query)
    print(tags)
