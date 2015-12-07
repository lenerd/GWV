#!/usr/bin/env python3

import datetime
import gzip
import pickle
import random
from collections import defaultdict, deque
import os
import sys


def main(start, length, last_n):
    """ Generates ten word sequences of length length starting with start.

        Require gzipped source file.
    """
    sourcefile = './heiseticker-text.txt.gz'
    cachefile = './mc-{}'.format(last_n)

    try:
        with gzip.open(cachefile, "rb") as f:
            P = pickle.load(f)
        print("# found cached marcov chain")
    except:
        print("# generating marcov chain")
        P = defaultdict(lambda: defaultdict(int))
        try:
            with gzip.open(sourcefile, "rt") as f:
                last = deque()
                for i in range(last_n):
                    last.append(f.readline().strip().lower())
                # count absolute frequencies
                for l in f:
                    word = l.strip().lower()
                    P[tuple(last)][word] += 1
                    last.popleft()
                    last.append(word)
        except:
            print("gzipped file '{}' not found".format(sourcefile))
        # calculate relative frequencies
        for f in P:
            n = sum(P[f].values())
            for s in P[f]:
                P[f][s] /= n

        with gzip.open(cachefile, "wb") as f:
            pickle.dump(dict(P), f)


    def choose(d):
        """ Wählt zufälligen Key aus d entsprechend der als Value angegebenen
        Wahrscheinlichkeit."""
        x = random.random()
        y = 0
        for k, v in sorted(d.items()):
            y += v
            if y >= x:
                return k

    if not list(filter(lambda x: x[-1] == start, P)):
        print("start word not found", file=sys.stderr)
        sys.exit(1)


    random.seed(datetime.datetime.now())

    for n in range(10):
        words = [start]

        # wähle mögliche Vergangenheit
        pos_hist = list(filter(lambda x: x[-1] == start, P))
        hist = random.choice(pos_hist)

        # wähle Worte anhand der gewählten Vergangenheit
        for i in range(last_n, 1, -1):
            words.append(choose(P[hist[-i:-1] + tuple(words)]))

        # wähle restliche Worte
        for i in range(length-last_n):
            words.append(choose(P[tuple(words[-last_n:])]))
        print(' '.join(words))
        print()



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: ./ticker.py <start> <length> <last-n>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1].lower(), int(sys.argv[2]), int(sys.argv[3]))
