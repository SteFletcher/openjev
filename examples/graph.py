"""A deliberately tiny graph runner, so the examples show the shape without a framework.

Nodes do work and return the updated state. Edges are routers: they look at the state and
name the next node. In LangGraph these are nodes and conditional edges, in Google ADK
agents and transfer rules; the System One reads sit on the edges in both.
"""

END = "__end__"


class Graph:
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.routers = {}

    def node(self, fn):
        self.nodes[fn.__name__] = fn
        return fn

    def edge(self, src, router):
        """router(state) -> next node name, or END. A plain string is a fixed edge."""
        self.routers[src] = router if callable(router) else (lambda _s, nxt=router: nxt)

    def run(self, start, state, max_steps=25):
        state.setdefault("trace", [])
        current = start
        for _ in range(max_steps):
            state = self.nodes[current](state) or state
            nxt = self.routers.get(current, lambda _s: END)(state)
            state["trace"].append(current if nxt == END else f"{current} -> {nxt}")
            if nxt == END:
                return state
            current = nxt
        raise RuntimeError(f"{self.name}: no END after {max_steps} steps: {state['trace']}")


def path(state):
    """The nodes a run visited, in order: generate -> judge -> deliver."""
    return " -> ".join(t.split(" -> ")[0] for t in state["trace"])
