#!/usr/bin/env python3

import itertools as it


class Constraint:
    def __init__(self, func, variables):
        """Creates a constraint object.

        Input:
            func       Function taking n arguments an returning a bool.
            variables  n-tuple of the affected variables.
        """
        self.func = func
        self.variables = variables

    def scope(self):
        """Returns the names of the affected variable.
        """
        return self.variables

    def __call__(self, *args):
        """Applies the constraint to the arguments."""
        return self.func(*args)


def arc_consistency(V, dom, C):
    """Implementation of the generalized arc consistency algorithm.

    Input
        V    a set of variables
        dom  a function such that dom(X) is the domain of variable X
        C    set of constraints to be satisfied
    Output
        arc-consistent domains for each variable X

    From Pool, Mackworth - Artifical Intelligence Foundation of Computational
    Agent, Figure 4.3.
    """

    def possible_assignments(X, x, c):
        """Possible assignments of the arguments to c with X fixed to x."""
        return {p for p in it.product(*(D_X[Y] if Y != X else {x}
                                        for Y in c.scope()))
                if c(*p)}

    D_X = {X: dom(X) for X in V}                  # initial variable domains
    TDA = {(X, c) for c in C for X in c.scope()}  # set of arcs to be checked

    while TDA:
        X, c = TDA.pop()  # choose an arc

        # new domain consisting of all values with assignments meeting the
        # constraint c
        ND_X = {x for x in D_X[X] if possible_assignments(X, x, c)}

        if D_X[X] != ND_X:  # the domain of X has changed

            # update the arcs to be checked with all arcs between other
            # variables and other constraints adjacent to X
            TDA = TDA.union({(Z, c_)
                            for c_ in C-{c} if X in c_.scope()
                            for Z in c_.scope() if Z != X})
            D_X[X] = ND_X

    return D_X  # reduced arc-consistent domains


def main():
    words = {'add', 'and', 'art', 'bag', 'far', 'ado', 'any', 'ash', 'ban',
             'fat', 'age', 'ape', 'ask', 'bat', 'fit', 'ago', 'apt', 'auk',
             'bee', 'lee', 'aid', 'arc', 'awe', 'boa', 'oaf', 'ail', 'are',
             'awl', 'ear', 'rat', 'aim', 'ark', 'aye', 'eel', 'tar', 'air',
             'arm', 'bad', 'eft', 'tie'}

    def dom(X):
        return words.copy()
    V = {'a1', 'a2', 'a3', 'd1', 'd2', 'd3'}
    C = {
        Constraint(lambda x, y: x[0] == y[0], ('a1', 'd1')),
        Constraint(lambda x, y: x[1] == y[0], ('a1', 'd2')),
        Constraint(lambda x, y: x[2] == y[0], ('a1', 'd3')),
        Constraint(lambda x, y: x[0] == y[1], ('a2', 'd1')),
        Constraint(lambda x, y: x[1] == y[1], ('a2', 'd2')),
        Constraint(lambda x, y: x[2] == y[1], ('a2', 'd3')),
        Constraint(lambda x, y: x[0] == y[2], ('a3', 'd1')),
        Constraint(lambda x, y: x[1] == y[2], ('a3', 'd2')),
        Constraint(lambda x, y: x[2] == y[2], ('a3', 'd3')),
    }
    ac_doms = arc_consistency(V, dom, C)
    print(ac_doms)


if __name__ == '__main__':
    main()
