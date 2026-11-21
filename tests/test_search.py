import random
import pytest

from puzzle.state import State, GOAL_STATE
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.a_star import AStar


def is_solvable(tiles):
    seq = [t for t in tiles if t != 0]
    inv = sum(1 for i in range(8) for j in range(i + 1, 8) if seq[i] > seq[j])
    return inv % 2 == 0


def scramble(moves, seed):
    rng = random.Random(seed)
    s = State(GOAL_STATE)
    for _ in range(moves):
        s = rng.choice(s.neighbors())
    return State(s.tiles)  # sem pai/custo


def replay(initial, actions):
    s = initial
    for a in actions:
        s = next(c for c in s.neighbors() if c.action == a)
    return s


def test_neighbors_counts():
    assert len(State((1, 2, 3, 4, 0, 5, 6, 7, 8)).neighbors()) == 4   # centro
    assert len(State((0, 1, 2, 3, 4, 5, 6, 7, 8)).neighbors()) == 2   # canto
    assert len(State((1, 0, 2, 3, 4, 5, 6, 7, 8)).neighbors()) == 3   # borda


def test_path_and_actions():
    s = State((1, 2, 3, 4, 5, 6, 7, 0, 8))
    child = s.neighbors()[0]
    grand = child.neighbors()[0]
    assert grand.path()[0] is s and grand.path()[-1] is grand
    assert len(grand.actions()) == 2
    assert s.actions() == []


@pytest.mark.parametrize("algo", [BFS(), DFS(), AStar()])
def test_already_goal(algo):
    r = algo.search(State(GOAL_STATE))
    assert r.found and r.path_cost == 0 and r.actions == []


@pytest.mark.parametrize("seed", range(5))
def test_optimality_and_validity(seed):
    initial = scramble(14, seed)
    bfs, astar, dfs = BFS().search(initial), AStar().search(initial), DFS().search(initial)
    assert bfs.found and astar.found and dfs.found
    assert bfs.path_cost == astar.path_cost          # ambos ótimos
    assert dfs.path_cost >= bfs.path_cost            # DFS pode ser pior
    for r in (bfs, astar, dfs):
        assert replay(initial, r.actions).tiles == GOAL_STATE
        assert r.depth == r.path_cost == len(r.actions)
    assert astar.nodes_expanded <= bfs.nodes_expanded


def test_unsolvable_returns_not_found():
    initial = State((1, 2, 3, 4, 5, 6, 8, 7, 0))
    assert not is_solvable(initial.tiles)
    for algo in (BFS(), AStar()):
        r = algo.search(initial)
        assert not r.found and r.actions == [] and r.path_cost == -1


def test_dfs_respects_depth_limit():
    initial = scramble(14, 0)
    assert not DFS(depth_limit=1).search(initial).found


def test_heuristic_admissible():
    h = AStar().heuristic
    assert h(State(GOAL_STATE)) == 0
    for seed in range(5):
        s = scramble(12, seed)
        assert h(s) <= BFS().search(s).path_cost
