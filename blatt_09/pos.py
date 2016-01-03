#!/usr/bin/env python3

import os
import sys
from collections import defaultdict
import pickle
import pprint


def train(input_files):
    init = defaultdict(int)
    trans = defaultdict(lambda: defaultdict(int))
    emiss = defaultdict(lambda: defaultdict(int))
    for file in input_files:
        with open(file, 'r') as f:
            line = f.readline()
            word, tag = line.rsplit(maxsplit=1)
            init[tag] += 1
            emiss[tag][word] += 1
            last = tag
            # count absolute frequencies
            for line in f:
                if not line.strip():
                    continue
                word, tag = line.rsplit(maxsplit=1)
                init[tag] += 1
                trans[last][tag] += 1
                emiss[tag][word] += 1
                last = tag

    # calculate relative frequencies
    n = sum(init.values())
    for t in init:
        init[t] /= n
    init = dict(init)
    for t1 in trans:
        n = sum(trans[t1].values())
        for t2 in trans[t1]:
            trans[t1][t2] /= n
        trans[t1] = dict(trans[t1])
    trans = dict(trans)
    for t in emiss:
        n = sum(emiss[t].values())
        for em in emiss[t]:
            emiss[t][em] /= n
        emiss[t1] = dict(emiss[t1])
    emiss = dict(emiss)

    with open("./.hmm-cache", "wb") as f:
        pickle.dump((init, trans, emiss), f)

    print("saved training data")


def _filter(obs, init, trans, emiss):
    if not obs:
        return []
    tags = []
    ob = obs[0]
    alpha = {s: init[s] * emiss[s].get(ob, 0)  for s in init}
    tags.append(max(alpha, key=lambda x:alpha[x]))
    for ob in obs[1:]:
        alpha = {s: emiss[s].get(ob, 0) * sum(alpha[q]*trans[q].get(s, 0) for q in init)
                 for s in init}

        tags.append(max(alpha, key=lambda x:alpha[x]))
    return tags


def viterbi(obs, init, trans, emiss):
    if not obs:
        return []
    tags = []
    ob = obs[0]

    delta = {s: init[s] * emiss[s].get(ob, 0) for s in init}
    if max(delta.values()) == 0:
        print("no state with emission: '{}'".format(ob))
        delta = {s: init[s] for s in init}

    pre = [{s: None for s in init}]
    for ob in obs[1:]:
        mx = {s: max(((q, delta[q] * trans[q].get(s, 0)) for q in init), key=lambda x: x[1])
              for s in init}
        for s in mx:
            mx[s] = (mx[s][0], mx[s][1] * emiss[s].get(ob, 0))
        if max(mx.values(), key=lambda x: x[1])[1] == 0:
            print("no transition to state with emission: '{}'".format(ob))
            mx = {s: max(((q, delta[q] * trans[q].get(s, 0)) for q in init),
                         key=lambda x: x[1])
                  for s in init}

        delta = {s: mx[s][1] for s in init}
        pre.append({s: mx[s][0] for s in init})
    t = max(delta, key=lambda x: delta[x])
    tags.insert(0, t)
    for p in pre[:0:-1]:
        t = p[tags[0]]
        tags.insert(0, t)
    return tags


def tag(input_file):

    if not os.path.isfile("./.hmm-cache") or not os.access("./.hmm-cache", os.R_OK):
        print("no training data found", file=sys.stderr)
        sys.exit(1)

    with open("./.hmm-cache", "rb") as f:
        init, trans, emiss = pickle.load(f)

    # words = input("> ").split()
    words = input_file.split()
    tags_f = _filter(words, init, trans, emiss)
    tags_v = viterbi(words, init, trans, emiss)
    print([(w, t1, t2) for w, t1, t2 in zip(words, tags_f, tags_v)])


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ['train', 'tag']:
        print("usage: ./pos.py train <input files>", file=sys.stderr)
        print("or:    ./pos.py tag   <input file>", file=sys.stderr)
        sys.exit(1)
    if sys.argv[1] == 'train':
        train(sys.argv[2:])
    else:
        tag(sys.argv[2])
