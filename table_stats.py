import json
import numpy as np
import heapq
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table_file',
        type=str,
        help="Path to JSON file containing the concept's table"
    )
    parser.add_argument(
        '--top_k',
        type=int,
        help='Top-k concepts to return (in terms of number of related object categories)'
    )
    args = parser.parse_args()

    with open(args.table_file) as fp:
        table = json.load(fp)
    more         = {k: v for k, v in table.items() if len(v) > 1}
    top_concepts = heapq.nlargest(n=args.top_k, iterable=more, key= lambda k: len(more[k]))
    top_concepts = {c: len(table[c]) for c in top_concepts}
    lens         = sum(len(v) for v in table.values())
    print("lens: ", lens)


    print("TABLE SIZE: {}".format(len(table)))
    print("CONCEPTS ASSOCIATED WITH MORE THAN ONE CLASS: {}".format(len(more)))
    print()
    print("TOP-{} CONCEPTS: {}".format(args.top_k, top_concepts))