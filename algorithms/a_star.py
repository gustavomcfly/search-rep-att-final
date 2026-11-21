import heapq
from itertools import count
from puzzle.base_search import BaseSearch
from puzzle.state import State, GOAL_STATE
from puzzle.result import SearchResult

# posição (linha, coluna) de cada peça no estado objetivo
_GOAL_POS = {tile: divmod(i, 3) for i, tile in enumerate(GOAL_STATE)}


class AStar(BaseSearch):
    """Busca A* com f(n) = g(n) + h(n)."""

    def heuristic(self, state: State) -> int:
        """Soma das distâncias de Manhattan de cada peça (exceto o vazio) até o objetivo.

        É admissível (nunca superestima o custo real, pois cada movimento desloca
        uma peça em uma casa) e consistente, o que garante solução ótima.
        """
        total = 0
        for index, tile in enumerate(state.tiles):
            if tile == 0:
                continue
            row, col = divmod(index, 3)
            goal_row, goal_col = _GOAL_POS[tile]
            total += abs(row - goal_row) + abs(col - goal_col)
        return total

    def search(self, initial: State) -> SearchResult:
        nodes_expanded = 0
        nodes_generated = 1
        tie = count()  # desempate estável no heap (evita comparar States)

        frontier = [(self.heuristic(initial), next(tie), initial)]
        best_g = {initial.tiles: initial.cost}
        max_frontier = 1

        while frontier:
            _, _, current = heapq.heappop(frontier)

            # entrada obsoleta: já existe caminho melhor até este estado
            if best_g.get(current.tiles, current.cost) < current.cost:
                continue

            if current.is_goal:
                return SearchResult(current, nodes_expanded, nodes_generated,
                                    max_frontier, current.cost)

            nodes_expanded += 1

            for child in current.neighbors():
                known = best_g.get(child.tiles)
                if known is not None and known <= child.cost:
                    continue
                best_g[child.tiles] = child.cost
                nodes_generated += 1
                heapq.heappush(frontier, (child.cost + self.heuristic(child), next(tie), child))

            max_frontier = max(max_frontier, len(frontier))

        return SearchResult(None, nodes_expanded, nodes_generated, max_frontier, 0)
