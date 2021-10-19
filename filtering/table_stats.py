import json
import numpy as np
import heapq

K=100

with open("concepts_table.json") as fp:
    table = json.load(fp)
more = {k: v for k, v in table.items() if len(v) > 1}
top_concepts = heapq.nlargest(n=K, iterable=more, key= lambda k: len(more[k]))
top_concepts = {c: len(table[c]) for c in top_concepts}
lens = sum(len(v) for v in table.values())
print("lens: ", lens)



print("TABLE SIZE: {}".format(len(table)))
print("CONCEPTS ASSOCIATED WITH MORE THAN ONE CLASS: {}".format(len(more)))
print()
print("TOP-{} CONCEPTS: {}".format(K, top_concepts))