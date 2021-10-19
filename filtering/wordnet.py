import json
import pdb

from nltk.corpus import wordnet as wn

"""concept table

Mapping from category (up to 80) to a set of related concepts (hypernims + hyponims).

"""


with open('data/MSCOCO/annotations/instances_val2017.json') as fp:
    categories = json.load(fp)['categories']
# synset.lemma_names("person.n.01") = ['person', 'individual', 'someone', 'somebody', 'mortal', 'soul']
conceptsTable = {}
for cat in categories:
    synset = wn.synset('{}.n.01'.format(cat['name']))
    concepts = set()
    pdb.set_trace()
    # {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
    #concepts[cat['name']] = concepts
