"""Part C: value iteration solver for the Part A map and the extended map.

Run:  python run_part_c.py
Everything is saved to outputs/part_c/ (the full printout is run_log.txt).
"""
import csv
import os
import random

from gridworld import ACTIONS, build_world, load_config
from value_iteration import extract_policy, q_value, value_iteration

OUT = os.path.join("outputs", "part_c")
os.makedirs(OUT, exist_ok=True)
ARROWS = {"up": "^", "right": ">", "down": "v", "left": "<"}
log_lines = []


def say(text=""):
    print(text)
    log_lines.append(text)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def value_map(world, V, digits=4):
    lines = []
    for r in range(1, world.n_rows + 1):
        cells = []
        for c in range(1, world.n_cols + 1):
            p = (r, c)
            if world.is_wall(p):
                cells.append("#".center(10))
            elif world.is_terminal(p):
                cells.append(world.letter(p).center(10))
            else:
                cells.append(("%.*f" % (digits, V[p])).rjust(10))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def policy_map(world, policy):
    lines = []
    for r in range(1, world.n_rows + 1):
        cells = []
        for c in range(1, world.n_cols + 1):
            p = (r, c)
            if world.is_wall(p) or world.is_terminal(p):
                cells.append(world.letter(p))
            else:
                cells.append(ARROWS[policy[p]])
        lines.append("  ".join(cells))
    return "\n".join(lines)


# ----------------------------------------------------------------- sanity checks
def run_checks(paper, extended):
    say("=== SANITY CHECKS (before trusting the solver) ===")
    n_ok = 0

    def check(name, condition, detail=""):
        nonlocal n_ok
        assert condition, "CHECK FAILED: " + name + " " + detail
        n_ok += 1
        say("PASS  " + name + (("  " + detail) if detail else ""))

    t = paper.transitions((1, 1), "up")
    check("top edge: moving up from (1,1) stays in place, reward of the tile it stays on",
          t == [(1.0, (1, 1), -1.0, "up")], str(t))
    t = paper.transitions((3, 4), "right")
    check("right edge: moving right from (3,4) stays in place",
          t == [(1.0, (3, 4), -1.0, "right")], str(t))
    t = paper.transitions((2, 1), "right")
    check("wall: moving right from (2,1) into the wall at (2,2) stays in place",
          t == [(1.0, (2, 1), -1.0, "right")], str(t))
    t = paper.transitions((1, 2), "down")
    check("wall on mud: moving down from mud (1,2) stays and pays the mud reward -3",
          t == [(1.0, (1, 2), -3.0, "down")], str(t))

    for a in ACTIONS:
        t = paper.transitions((2, 3), a)
        total = sum(p for p, _, _, _ in t)
        check("ice at (2,3), action %s: outcome probabilities sum to 1" % a,
              abs(total - 1.0) < 1e-12 and len(t) == 3, "outcomes=%s" % [(round(p, 2), n, r) for p, n, r, _ in t])
    t = paper.transitions((2, 3), "up")
    check("ice up: 0.8 to (1,3), 0.1 blocked by wall stays (2,3), 0.1 to hole (2,4)",
          sorted(t) == sorted([(0.8, (1, 3), -1.0, "up"), (0.1, (2, 3), -1.0, "left"),
                               (0.1, (2, 4), -10.0, "right")]), str(t))
    t = paper.transitions((1, 3), "down")
    check("slipping depends on the tile you LEAVE: moving onto ice from (1,3) is deterministic",
          t == [(1.0, (2, 3), -1.0, "down")], str(t))

    check("mud reward: entering mud (1,2) from (1,1) gives -3",
          paper.transitions((1, 1), "right")[0][2] == -3.0)
    check("sand reward: entering sand (3,2) from (3,1) gives -2 (extended map)",
          extended.letter((3, 2)) == "D" and extended.transitions((3, 1), "right")[0][2] == -2.0)
    check("water reward: entering water (3,4) from (3,3) gives -5 (extended map)",
          extended.letter((3, 4)) == "W" and extended.transitions((3, 3), "right")[0][2] == -5.0)
    check("sand and water are deterministic",
          len(extended.transitions((3, 2), "up")) == 1 and len(extended.transitions((3, 4), "left")) == 1)
    check("goal reward: entering G gives +10 once, no extra floor cost",
          paper.transitions((1, 3), "right") == [(1.0, (1, 4), 10.0, "right")])
    check("hole reward: entering H gives -10",
          paper.transitions((3, 4), "up") == [(1.0, (2, 4), -10.0, "up")])

    for name, w in (("paper map", paper), ("extended map", extended)):
        for term in [p for p in w.states() if w.is_terminal(p)]:
            check("%s: terminal %s (%s) has no transitions, so no rewards after termination"
                  % (name, term, w.letter(term)),
                  all(w.transitions(term, a) == [] for a in ACTIONS))
        check("%s: wall is not a state" % name,
              all(not w.is_wall(p) for p in w.states()) and (2, 2) not in w.states())
    rollout_reward, steps = 0.0, 0
    pos = (1, 3)
    while not paper.is_terminal(pos):                 # one move into the goal
        _, pos, r, _ = paper.transitions(pos, "right")[0]
        rollout_reward += r
        steps += 1
    check("rollout (1,3) -> right: total reward exactly +10 in 1 step, nothing after termination",
          rollout_reward == 10.0 and steps == 1)
    say("%d checks passed.\n" % n_ok)


# --------------------------------------------------------------------- solving
def solve(name, world, cfg, paper_check=False):
    say("=" * 70)
    say("MAP: %s" % name)
    say("\n".join(world.rows))
    sol = cfg["solver"]
    V, iters, change, converged, history = value_iteration(
        world, sol["tolerance"], sol["max_iterations"], cfg.get("initial_value", 0))
    if paper_check:
        say("\nV1 (compare with the paper):")
        say(value_map(world, history[1]))
        expected_v1 = {p: (10.0 if p == (1, 3) else -1.0) for p in world.nonterminal_states()}
        diff1 = max(abs(history[1][p] - expected_v1[p]) for p in expected_v1)
        say("largest difference from my paper V1: %.2e" % diff1)
        say("\nV2 at the three requested positions (action values from V1):")
        paper_v2 = {(1, 2): 8.0, (1, 3): 10.0, (2, 3): 5.21}
        for pos in map(tuple, cfg["paper_v2_positions"]):
            q = {a: q_value(world, history[1], pos, a) for a in ACTIONS}
            best = max(q, key=q.get)
            say("  %s: up=%.4f right=%.4f down=%.4f left=%.4f -> V2=%.4f (best: %s); paper %.4f, "
                "difference %.2e" % (pos, q["up"], q["right"], q["down"], q["left"],
                                     history[2][pos], best, paper_v2[pos], abs(history[2][pos] - paper_v2[pos])))
    say("\nvalue iteration: %d iterations, final largest change %.3e, converged=%s"
        % (iters, change, converged))
    if not converged:
        say("WARNING: reached the %d-iteration safeguard without converging." % sol["max_iterations"])
    policy, qs, ties = extract_policy(world, V)
    say("\nFinal values:")
    say(value_map(world, V))
    say("\nPolicy (ties broken by the order up, right, down, left):")
    say(policy_map(world, policy))
    tied = {p: t for p, t in ties.items() if len(t) > 1}
    say("States with tied best actions: %s" % (tied if tied else "none"))
    write_csv("%s_values_policy.csv" % name,
              ["row", "col", "tile", "value", "chosen_action", "tied_actions",
               "Q_up", "Q_right", "Q_down", "Q_left"],
              [[p[0], p[1], world.letter(p), V[p], policy[p], "|".join(ties[p])] +
               [qs[p][a] for a in ACTIONS] for p in sorted(V)])
    write_csv("%s_iterations.csv" % name, ["iteration"] + ["%d,%d" % p for p in sorted(V)],
              [[k] + [h[p] for p in sorted(V)] for k, h in enumerate(history)])
    return V, policy, ties, (iters, change, converged)


# ------------------------------------------------------------------ simulation
def simulate_episode(world, policy, episode_seed, max_steps):
    rng = random.Random(episode_seed)
    pos, steps, ret, disc = world.start(), 0, 0.0, 1.0
    while steps < max_steps and not world.is_terminal(pos):
        outcomes = world.transitions(pos, policy[pos])
        u, acc, chosen = rng.random(), 0.0, outcomes[-1]
        for o in outcomes:
            acc += o[0]
            if u < acc:
                chosen = o
                break
        _, pos, reward, _ = chosen
        ret += disc * reward
        disc *= world.discount
        steps += 1
    if world.is_terminal(pos):
        outcome = world.tiles[world.letter(pos)]["meaning"]        # "goal" or "hole"
    else:
        outcome = "time_limit"
    return steps, ret, outcome


def simulate(name, world, policy, V, cfg):
    sim = cfg["simulation"]
    ep_rows, batch_rows = [], []
    for bs in sim["batch_seeds"]:
        counts = {"goal": 0, "hole": 0, "time_limit": 0}
        steps_sum = ret_sum = 0.0
        for i in range(sim["episodes_per_batch"]):
            seed = 1000 * bs + i
            steps, ret, outcome = simulate_episode(world, policy, seed, sim["max_steps"])
            counts[outcome] += 1
            steps_sum += steps
            ret_sum += ret
            ep_rows.append([name, bs, seed, steps, ret, outcome])
        n = sim["episodes_per_batch"]
        batch_rows.append([name, bs, counts["goal"], counts["hole"], counts["time_limit"],
                           steps_sum / n, ret_sum / n])
    write_csv("%s_simulation_episodes.csv" % name,
              ["map", "batch_seed", "episode_seed", "steps", "discounted_return", "outcome"], ep_rows)
    write_csv("%s_simulation_batches.csv" % name,
              ["map", "batch_seed", "goals_of_50", "holes_of_50", "time_limits_of_50",
               "mean_steps", "mean_discounted_return"], batch_rows)
    say("\nSimulation of the computed policy on %s (3 batches x 50 episodes, max %d steps):"
        % (name, sim["max_steps"]))
    say("| Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |")
    say("| --- | --- | --- | --- | --- | --- |")
    for r in batch_rows:
        say("| %d | %d | %d | %d | %.2f | %.4f |" % (r[1], r[2], r[3], r[4], r[5], r[6]))
    overall = sum(r[6] for r in batch_rows) / len(batch_rows)
    say("mean discounted return over all 150 episodes: %.4f   value of the start tile V*(S) = %.4f"
        % (overall, V[world.start()]))
    return batch_rows, overall


def main():
    cfg = load_config()
    paper = build_world(cfg, "paper_map")
    extended = build_world(cfg, "extended_map")
    run_checks(paper, extended)
    V_p, pol_p, ties_p, info_p = solve("paper_map", paper, cfg, paper_check=True)
    sim_p = simulate("paper_map", paper, pol_p, V_p, cfg)
    V_e, pol_e, ties_e, info_e = solve("extended_map", extended, cfg)
    sim_e = simulate("extended_map", extended, pol_e, V_e, cfg)

    say("\n" + "=" * 70)
    say("COMPARISON: paper map vs extended map (same coordinates)")
    rows = []
    say("tile changes: " + ", ".join("%s %s->%s" % (p, paper.letter(p), extended.letter(p))
                                     for p in paper.states() if paper.letter(p) != extended.letter(p)))
    say("(row, col) | tile paper->ext | V paper | V ext | change | action paper | action ext")
    for p in sorted(V_p):
        d = V_e[p] - V_p[p]
        changed = "CHANGED" if pol_p[p] != pol_e[p] else ""
        rows.append([p[0], p[1], paper.letter(p), extended.letter(p), V_p[p], V_e[p], d, pol_p[p], pol_e[p],
                     bool(changed)])
        say("%s | %s->%s | %9.4f | %9.4f | %+9.4f | %-5s | %-5s %s" % (
            p, paper.letter(p), extended.letter(p), V_p[p], V_e[p], d, pol_p[p], pol_e[p], changed))
    write_csv("comparison.csv", ["row", "col", "tile_paper", "tile_extended", "V_paper", "V_extended",
                                 "V_change", "action_paper", "action_extended", "action_changed"], rows)
    ice = (2, 3)
    say("\nIce tile (2,3): paper action %s, extended action %s; Q-values (paper): %s"
        % (pol_p[ice], pol_e[ice],
           {a: round(q_value(paper, V_p, ice, a), 4) for a in ACTIONS}))
    say("Iterations: paper map %d (change %.2e, converged %s); extended map %d (change %.2e, converged %s)"
        % (info_p[0], info_p[1], info_p[2], info_e[0], info_e[1], info_e[2]))
    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
