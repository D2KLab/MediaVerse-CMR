import json
import pdb
from typing import Dict, List, Set

import requests
import spacy
from tqdm import tqdm

LANG         = 'en'
SCHEMA       = 'http://api.conceptnet.io/c/{}/{}?offset={}&limit={}'
TO_IGNORE    = set(['ADP', 'NUM', 'SYM', 'ADV', 'PRON', 'CCONJ', 'SCONJ', 'AUX', 'DET', 'PART', 'PUNCT', 'INTJ']) # universal POS tags: https://universaldependencies.org/u/pos/
nlp          = spacy.load("en_core_web_sm")

def add_concepts(edges: List[Dict], table: Dict[str, Set], tag: str) -> None:
    #edges[0]['surfaceText'] = [[a handbag]] is for [[carrying makeup]]
    #   idea: splitting the end into tokens "carrying" + "makeup"
    #   then filter out concepts connected to more than 50% of categories (not discriminative)
    #       - remove articles (the, a, ...)
    #       - remove "is a translation of" (take only edges[i]['start']['language'] == 'en' and edges[i]['end']['language'])
    #       - remove "ExternalURL" relation (in this case the edge['end'] does not containg the 'language' key)
    for edge in edges:
        if edge['start']['language'] != 'en' or 'language' not in edge['end'] or edge['end']['language'] != 'en':
            continue
        doc = nlp(edge['end']['label'].lower())     
        for token in doc:
            concept = token.lemma_
            if len(token.text) == 1 or token.pos_ in TO_IGNORE:
                continue
            if concept in table:
                table[concept].add(tag)
            else:
                table[concept] = {tag}


if __name__ == '__main__':
    with open('data/MSCOCO/annotations/instances_val2017.json') as fp:
        categories = json.load(fp)['categories']   
    table = {}
    cats = set([cat['name'].lower() for cat in categories])
    print(cats)
    bar = tqdm(cats)
    for cat in bar:
        #each page contains up to 20 relations for that node. To see more go to next page using obj['view']['nextPage']
        #concept = 'sand'
        bar.set_description(cat)
        offset  = 0
        limit   = 20
        while True:
            # if more than 1 word then format it as <first_word>_<second_word>
            curr_tag = cat if cat.split(' ') == 1 else '_'.join(cat.split(' '))
            obj     = requests.get(SCHEMA.format(LANG, curr_tag, offset, limit)).json()
            if 'error' in obj:
                exit('Error in request with {}'.format(cat))
            add_concepts(obj['edges'], table, cat)
            if 'view' not in obj or 'nextPage' not in obj['view']:
                break
            next_url = obj['view']['nextPage']
            params = next_url.split('?')[-1]
            offset, limit = params.split('&')[0].split('=')[-1], params.split('&')[1].split('=')[-1]

    # turn sets into lists (set is not JSON serializable)
    for k,v in table.items():
        table[k] = list(v)
    with open('concepts_table.json', 'w') as fp:
        json.dump(table, fp)

