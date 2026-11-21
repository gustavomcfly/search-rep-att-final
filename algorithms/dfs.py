from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult

DEFAULT_DEPTH_LIMIT = 50


class DFS(BaseSearch):
    """Busca em Profundidade com limite de profundidade (pilha LIFO).

    O limite evita que a busca desça indefinidamente. Para não perder soluções
    dentro do limite, um estado só é descartado se já foi alcançado em
    profundidade menor ou igual; se aparece de novo mais raso, é reexplorado.
    A solução encontrada NÃO é necessariamente ótima.
    """

    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT):
        self.depth_limit = depth_limit

    def search(self, initial: State) -> SearchResult:
        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1

        stack = [initial]
        best_depth = {initial.tiles: initial.cost}

        while stack:
            current = stack.pop()

            # entrada obsoleta: o estado já foi alcançado por um caminho mais raso
            if best_depth.get(current.tiles, current.cost) < current.cost:
                continue

            if current.is_goal:
                return SearchResult(current, nodes_expanded, nodes_generated,
                                    max_frontier, current.cost)

            if current.cost >= self.depth_limit:
                continue

            nodes_expanded += 1

            # inverte para que o primeiro filho seja o primeiro a ser explorado
            for child in reversed(current.neighbors()):
                known = best_depth.get(child.tiles)
                if known is not None and known <= child.cost:
                    continue
                best_depth[child.tiles] = child.cost
                nodes_generated += 1
                stack.append(child)

            max_frontier = max(max_frontier, len(stack))

        return SearchResult(None, nodes_expanded, nodes_generated, max_frontier, 0)
