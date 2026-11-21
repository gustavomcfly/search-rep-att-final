from collections import deque
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class BFS(BaseSearch):
    """Busca em Largura (fila FIFO). Completa e ótima para custo uniforme."""

    def search(self, initial: State) -> SearchResult:
        nodes_expanded = 0
        nodes_generated = 1  # o estado inicial
        max_frontier = 1

        if initial.is_goal:
            return SearchResult(initial, nodes_expanded, nodes_generated, max_frontier, initial.cost)

        frontier = deque([initial])
        visited = {initial.tiles}  # estados já vistos (fronteira + expandidos)

        while frontier:
            current = frontier.popleft()
            nodes_expanded += 1

            for child in current.neighbors():
                if child.tiles in visited:
                    continue
                visited.add(child.tiles)
                nodes_generated += 1

                # teste de objetivo na geração: mais eficiente e continua ótimo na BFS
                if child.is_goal:
                    return SearchResult(child, nodes_expanded, nodes_generated,
                                        max(max_frontier, len(frontier) + 1), child.cost)
                frontier.append(child)

            max_frontier = max(max_frontier, len(frontier))

        return SearchResult(None, nodes_expanded, nodes_generated, max_frontier, 0)
