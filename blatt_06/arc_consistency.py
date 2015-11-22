#!/usr/bin/env python3

import itertools as it


words = {'add', 'and', 'art', 'bag', 'far', 'ado', 'any', 'ash', 'ban', 'fat',
         'age', 'ape', 'ask', 'bat', 'fit', 'ago', 'apt', 'auk', 'bee', 'lee',
         'aid', 'arc', 'awe', 'boa', 'oaf', 'ail', 'are', 'awl', 'ear', 'rat',
         'aim', 'ark', 'aye', 'eel', 'tar', 'air', 'arm', 'bad', 'eft', 'tie'}


def generalized_arc_consistency(V, dom, C):

    def scope(c):
        return c[1]

    def possible_assignments(X, x, c):
        return [p for p in it.product(*(D_X[Y] if Y != X else {x}
                                        for Y in scope(c)))
                if c[0](*p)]

    D_X = {X: dom(X) for X in V}
    TDA = {(X, c) for c in C for X in scope(c)}

    while TDA:
        X, c = TDA.pop()
        ND_X = set()
        for x in D_X[X]:
            if possible_assignments(X, x, c):
                ND_X.add(x)
        if D_X[X] != ND_X:
            TDA = TDA.union({(Z, c_)
                            for c_ in C-{c} if X in scope(c_)
                            for Z in scope(c_) if Z != X})
            D_X[X] = ND_X

    return D_X


if __name__ == '__main__':
    def dom(x):
        return words
    V = {'a1', 'a2', 'a3', 'd1', 'd2', 'd3'}
    C = {
        (lambda x, y: x[0] == y[0], ('a1', 'd1')),
        (lambda x, y: x[1] == y[0], ('a1', 'd2')),
        (lambda x, y: x[2] == y[0], ('a1', 'd3')),
        (lambda x, y: x[0] == y[1], ('a2', 'd1')),
        (lambda x, y: x[1] == y[1], ('a2', 'd2')),
        (lambda x, y: x[2] == y[1], ('a2', 'd3')),
        (lambda x, y: x[0] == y[2], ('a3', 'd1')),
        (lambda x, y: x[1] == y[2], ('a3', 'd2')),
        (lambda x, y: x[2] == y[2], ('a3', 'd3')),
    }
    ac_doms = generalized_arc_consistency(V, dom, C)
    print(ac_doms)
