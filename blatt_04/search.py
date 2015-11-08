#!/usr/bin/env python3
import numpy as np
import heapq
import itertools as it
import functools as fn
import sys


class Grid:
    """Represents a 2D-labyrinth."""

    def __init__(self, filename):
        """Loads a labyrinth from a given file.

        The file is expected to contain an ASCII labyrinth
        meeting the following conditions:
            - every line as the same length
            - it contains exact one start position
            - it has an impassable border of 'x's
            - TODO: portals
        """
        with open(filename, 'r') as f:
            self.data = np.array([list(l) for l in f.read().splitlines()])
        assert self.data.ndim == 2  # the grid is rectangular
        # start node
        tmp = np.where(self.data == 's')
        assert len(tmp[0]) != 0  # a start node exists
        self.start = (tmp[0][0], tmp[1][0])
        # goal node
        tmp = np.where(self.data == 'g')
        assert len(tmp[0]) != 0  # a goal node exists
        self.goal = (tmp[0][0], tmp[1][0])

        self.portals = set()
        self.portal_dst = {}
        for i in range(1, 10):
            tmp = np.where(self.data == str(i))
            if len(tmp[0]) != 2:
                break
            self.portals.add(((tmp[0][0], tmp[1][0]),
                                  (tmp[0][1], tmp[1][1])))
            self.portal_dst[(tmp[0][0], tmp[1][0])] = (tmp[0][1], tmp[1][1])
            self.portal_dst[(tmp[0][1], tmp[1][1])] = (tmp[0][0], tmp[1][0])

        return

        # TODO: calc heuristic
    def neighbours(self, pos):
        """Returns the adjacent and not blocked positions.

        If pos is next to a portal, its destination is considered as a
        neighbour."""
        r, c = pos
        n =  {self.portal_dst.get(n, n)
              for n in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
              if not self.is_blocked(n)}

        return n

    def is_goal(self, pos):
        """Returns whether a the given position is a goal."""
        return self.data[pos] == 'g'

    def is_blocked(self, pos):
        """Returns whether the given position is blocked."""
        return self.data[pos] == 'x'

    def __str__(self):
        """Represents the grid as an ASCII string."""
        return '\n'.join(''.join(r) for r in self.data)

    def print_path(self, path):
        """Draws the given path into the string representation."""
        return '\n'.join(''.join('.' if (rn, cn) in path and
                                 c not in 'xgs'
                                 else c
                                 for cn, c in enumerate(r))
                         for rn, r in enumerate(self.data))


def manhattan(pos1, pos2):
    """Calculates the manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def portal_routes(portal_pairs):
    """Return all possible routes through a set of two-way portals.

    Any pair of portals is not used more than once."""

    def swap(p):
        return p[1], p[0]

    routes = set()
    for l in range(len(portal_pairs) + 1):
        for p in it.permutations(portal_pairs, l):
            for s in it.product([0, 1], repeat=l):
                routes.add(tuple(pi if si else swap(pi)
                                 for pi, si in zip(p, s)))
    return routes


def portal_heuristic(grid):
    """Calculate a lower bound of the distance to grid.goal for each portal.

    Routes through portals are considered."""
    heuristics = {}
    pr = portal_routes(grid.portals)
    for p_start in grid.portal_dst:
        min_dist = manhattan(p_start, grid.goal)
        for route in filter(lambda x: len(x) and x[0][0] == p_start, pr):
            s = sum(manhattan(route[i][1], route[i+1][0]) for i in range(len(route)-1))
            s += manhattan(route[-1][1], grid.goal)
            min_dist = min(min_dist, s)
        heuristics[p_start] = min_dist
    return heuristics


def heuristic(pos, grid, p_heuristic):
    """Returns a lower bound of the distance from pos to grid.goal."""
    min_dist = manhattan(pos, grid.goal)
    for p_start in grid.portal_dst:
        min_dist = min(min_dist, manhattan(pos, p_start) +  p_heuristic[p_start])
    return min_dist


def a_star(grid, verbose=False):
    """Searches a path from the start to the goal node in grid.

    Parameter:
        grid    -- instance of Grid to work on
        verbose -- prints the current path on every step

    Returns:
        Cost, Path from start to a goal if one is found,
        else an empty list.
    """
    stats = {'time': 0, 'space': 0}
    p_heuristic = portal_heuristic(grid)
    marked = np.zeros(grid.data.shape, dtype=bool)
    marked[grid.start] = True
    # (estimate, cost, path)
    frontier = [(heuristic(grid.start, grid, p_heuristic), 0, [grid.start])]
    while frontier:
        stats['time'] += 1
        stats['space'] = max(stats['space'], sum(len(p) for p in frontier))
        _, cost, path = heapq.heappop(frontier)
        if verbose and False:
            print(grid.print_path(path))
        cost += 1
        for pos in grid.neighbours(path[-1]):
            if marked[pos]:
                continue
            if grid.is_goal(pos):
                return cost, path + [pos], stats
            else:
                marked[pos] = True
                heapq.heappush(frontier,
                               (heuristic(pos, grid, p_heuristic) + cost,
                                cost, path + [pos]))
    return float('inf'), [], stats


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Labyrinth as first argument.", file=sys.stderr)
        sys.exit(1)
    g = Grid(sys.argv[1])
    print("A* search")
    cost, p, stats = a_star(g, verbose=True)
    print(p)
    print("length: {}".format(cost))
    print("time:  {} iterations".format(stats['time']))
    print("space: {} nodes in the frontier".format(stats['space']))
