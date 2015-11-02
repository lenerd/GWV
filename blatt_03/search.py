#!/usr/bin/env python3
import numpy as np
from collections import deque


class Grid:
    """Represents a 2D-labyrinth."""

    def __init__(self, filename):
        """Loads a labyrinth from a given file.

        The file is expected to contain an ASCII labyrinth
        meeting the following conditions:
            - every line as the same length
            - it contains exact one start position
            - it has an impassable border of 'x's
        """
        with open(filename, 'r') as f:
            self.data = np.array([list(l) for l in f.read().splitlines()])
        assert self.data.ndim == 2  # the grid is rectangular
        tmp = np.where(self.data == 's')
        assert len(tmp[0]) != 0  # a start node exists
        self.start = (tmp[0][0], tmp[1][0])

    def neighbours(self, r, c):
        """Returns the adjacent and not blocked positions."""
        return [pos for pos in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                if not self.blocked(*pos)]

    def goal(self, r, c):
        """Returns whether a the given position is a goal."""
        return self.data[r][c] == 'g'

    def blocked(self, r, c):
        """Returns whether the given position is blocked."""
        return self.data[r][c] == 'x'

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


def search(grid, alg, verbose=False):
    """Starts a search for a goal from the start node.

    Parameter:
        grid    -- instance of Grid to work on
        alg     -- algorithm to use
                    'bfs' -> breadth-first search
                    'dfs' -> depth-first search
        verbose -- prints the current path on every step

    Returns:
        Path from start to a goal if one is found,
        else an empty list.
    """
    assert alg in ['bfs', 'dfs']
    marked = np.zeros(grid.data.shape, dtype=bool)
    marked[grid.start] = True
    frontier = deque([[grid.start]])
    # use the deque as stack or queue
    select = frontier.pop if alg == 'dfs' else frontier.popleft
    while frontier:  # is not empty
        path = select()
        if verbose:
            print(grid.print_path(path))
        for pos in grid.neighbours(*path[-1]):
            if marked[pos]:
                continue
            if grid.goal(*pos):
                return path + [pos]
            else:
                marked[pos] = True
                frontier.append(path + [pos])
    return []


if __name__ == '__main__':
    g = Grid('blatt3_environment.txt')
    print("Depth-first search")
    p = search(g, 'dfs', verbose=True)
    print(p)
    print("\n")
    print("Breadth-first search")
    q = search(g, 'bfs', verbose=True)
    print(q)
