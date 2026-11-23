"""Value iteration written from scratch, using the known transition model."""
from gridworld import ACTIONS


def q_value(world, V, pos, action):
    """sum over s' of P(s'|s,a) * (R + discount * V(s')); terminal V = terminal_value."""
    total = 0.0
    for prob, nxt, reward, _ in world.transitions(pos, action):
        v_next = world.terminal_value if world.is_terminal(nxt) else V[nxt]
        total += prob * (reward + world.discount * v_next)
    return total


def value_iteration(world, tolerance=0.000001, max_iterations=10000, initial_value=0.0):
    """Returns (V, iterations, last_change, converged, history).
    history[k] holds the values after k iterations (history[0] = V0).
    Each round uses ONLY the previous round's values."""
    V = {p: float(initial_value) for p in world.nonterminal_states()}
    history = [dict(V)]
    change = float("inf")
    for k in range(1, max_iterations + 1):
        new = {p: max(q_value(world, V, p, a) for a in ACTIONS) for p in V}
        change = max(abs(new[p] - V[p]) for p in V)
        V = new
        history.append(dict(V))
        if change < tolerance:
            return V, k, change, True, history
    return V, max_iterations, change, False, history


def extract_policy(world, V, tie_tolerance=1e-9):
    """Greedy policy from the final values.
    Actions within tie_tolerance of the best are tied; ties are broken by taking the first
    of ACTIONS = up, right, down, left. Returns (policy, q_values, ties)."""
    policy, qs, ties = {}, {}, {}
    for p in world.nonterminal_states():
        q = {a: q_value(world, V, p, a) for a in ACTIONS}
        best = max(q.values())
        tied = [a for a in ACTIONS if best - q[a] <= tie_tolerance]
        policy[p], qs[p], ties[p] = tied[0], q, tied
    return policy, qs, ties
