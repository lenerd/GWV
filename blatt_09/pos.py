#!/usr/bin/env python3

import re
import os
import sys
from collections import defaultdict
import pickle
import pprint


def train(input_files):
    """Trainiert das HMM.

    Daten werden in ./.hmm-cache gespeichert.
    """

    # initiale Wahrscheinlichkeiten
    init = defaultdict(int)
    # Übergangswahrscheinlichkeiten
    trans = defaultdict(lambda: defaultdict(int))
    # Emissionswahrscheinlichkeiten
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

    # saving probabilities
    with open("./.hmm-cache", "wb") as f:
        pickle.dump((init, trans, emiss), f)

    print("saved training data")


def _filter(obs, init, trans, emiss):
    """Tagt eine Liste von Wörtern durch Filtern.

    obs:    beobachtete Wörter
    init:   initiale Wahrscheinlichkeiten
    trans:  Übergangswahrscheinlichkeiten
    emiss:  Emissionswahrscheinlichkeiten

    Gibt Liste der Tags zurück.
    
    Implementiert den Forward-Algorithmus (Folien ch06.pdf, S. 135).
    Macht Unsinn bei unbekannten Wörtern.
    """

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
    """Tagt eine Liste von Wörtern mit dem Viterbi-Algorithmus

    obs:    beobachtete Wörter
    init:   initiale Wahrscheinlichkeiten
    trans:  Übergangswahrscheinlichkeiten
    emiss:  Emissionswahrscheinlichkeiten

    Gibt Liste der Tags zurück.
    
    Implementiert den Viterbi-Algorithmus (Folien ch06.pdf, S. 152).
    Macht weniger Unsinn bei unbekannten Wörtern.
    """
    if not obs:
        return []
    tags = []
    ob = obs[0]

    # delta_1(s) = init[s] * E[s,o_1]
    delta = {s: init[s] * emiss[s].get(ob, 0) for s in init}
    if max(delta.values()) == 0:
        # ignoriere Emissionswahrscheinlichkeiten bei unbekannten Wörtern
        delta = {s: init[s] for s in init}

    pre = [{s: None for s in init}]
    for ob in obs[1:]:
        # (argmax_q(delta_k(q) * T_q,s),
        #  max_q(delta_k(q) * T_q,s))
        mx = {s: max(((q, delta[q] * trans[q].get(s, 0)) for q in init),
                     key=lambda x: x[1])
              for s in init}
        # (argmax_q(delta_k(q) * T_q,s),
        #  max_q(delta_k(q) * T_q,s) * E_s,o_k+1)
        for s in mx:
            mx[s] = (mx[s][0], mx[s][1] * emiss[s].get(ob, 0))
        if max(mx.values(), key=lambda x: x[1])[1] == 0:
            # ignoriere Emissionswahrscheinlichkeiten bei unbekannten Wörtern
            mx = {s: max(((q, delta[q] * trans[q].get(s, 0)) for q in init),
                         key=lambda x: x[1])
                  for s in init}

        # delta_k+1(s) = max_q(...
        delta = {s: mx[s][1] for s in init}
        # pre_k+1(s) = argmax_q(...
        pre.append({s: mx[s][0] for s in init})

    # reconstruction
    t = max(delta, key=lambda x: delta[x])
    tags.insert(0, t)
    for p in pre[:0:-1]:
        t = p[tags[0]]
        tags.insert(0, t)
    return tags


def tag(words):
    """Tags the wordlist."""
    if not os.path.isfile("./.hmm-cache") or not os.access("./.hmm-cache", os.R_OK):
        print("no training data found", file=sys.stderr)
        sys.exit(1)

    with open("./.hmm-cache", "rb") as f:
        init, trans, emiss = pickle.load(f)

    tags_f = _filter(words, init, trans, emiss)
    tags_v = viterbi(words, init, trans, emiss)
    return [(w, t2, t1) for w, t1, t2 in zip(words, tags_f, tags_v)]


def split_sentence(s):
    """Splits s at whitespace and punctuations."""
    pattern = r'([.,;:-?!\"()\[\]{}])'
    s = re.sub(pattern, r" \1", s)
    return s.split()


def tag_input(inp):
    """Tags inp"""
    words = split_sentence(inp)
    tagged = tag(words)
    w_len = 20
    t_len = 10
    print("word" + " " * (w_len-len("word")) + \
          "viterbi" + " " * (t_len-len("viterbi")) + "forward")
    print("-"*47)
    for w, tv, tf in tag(words):
        print(w + " " * (w_len-len(w)) + \
              tv + " " * (t_len-len(tv)) + \
              tf)
    print()


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ['train', 'tag']:
        print("usage: ./pos.py train <input files>", file=sys.stderr)
        print("or:    ./pos.py tag   <sentence>", file=sys.stderr)
        sys.exit(1)
    if sys.argv[1] == 'train':
        train(sys.argv[2:])
    else:
        tag_input(sys.argv[2])
