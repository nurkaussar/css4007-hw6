"""Part B: simulate FrozenLake-v1 (4x4) with Gymnasium. No learning happens here.

Run:  python run_part_b.py
Compares uniformly random actions with a fixed policy designed from the map,
with is_slippery False and True: 4 conditions x 3 batches x 50 episodes.
Results go to outputs/part_b/.
"""
import csv
import os
import platform
import random
import re

import gymnasium as gym

MAP = ["SFFF",
       "FHFH",
       "FFFH",
       "HFFG"]
BATCH_SEEDS = (11, 22, 33)
EPISODES_PER_BATCH = 50
MAX_STEPS = 100
GOAL_STATE = 15
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3            # Gymnasium FrozenLake action codes
ACTION_NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}
OUT = os.path.join("outputs", "part_b")
os.makedirs(os.path.join(OUT, "traces"), exist_ok=True)

# State numbers (row * 4 + column):
#   0  1  2  3        S F F F
#   4  5  6  7        F H F H
#   8  9 10 11        F F F H
#  12 13 14 15        H F F G
# Fixed policy, designed from the map: take the shortest route that never steps on a hole
#   0 -down-> 4 -down-> 8 -right-> 9 -right-> 10 -down-> 14 -right-> 15 (goal).
# Going "down" from 8 would enter the hole at 12, and "down" from 1 or 7-side tiles would
# enter holes, so those states get the safe direction. One action for EVERY nonterminal
# state (holes 5, 7, 11, 12 and goal 15 are terminal).
FIXED_POLICY = {
    0: DOWN,    # start: go down the left edge
    1: RIGHT,   # (down would be the hole at 5)
    2: DOWN,    # toward the goal side
    3: LEFT,    # (down would be the hole at 7)
    4: DOWN,    # (right would be the hole at 5)
    6: DOWN,    # toward the bottom row
    8: RIGHT,   # (down would be the hole at 12)
    9: RIGHT,   # toward 10
    10: DOWN,   # toward 14
    13: RIGHT,  # toward 14
    14: RIGHT,  # into the goal
}
NONTERMINAL = [s for s in range(16) if MAP[s // 4][s % 4] not in "HG"]
assert sorted(FIXED_POLICY) == NONTERMINAL, "fixed policy must cover every nonterminal state"


def clean_render(text):
    """Text rendering: mark the agent's tile with [x] and drop terminal colour codes."""
    text = re.sub(r"\x1b\[[0-9;]*m(.)\x1b\[0m", r"[\1]", text)
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def make_env(slippery):
    # default rewards and slip probabilities: nothing is customised
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode="ansi")


def run_episode(env, choose_action, episode_seed, max_steps=MAX_STEPS):
    """One episode with reset()/step(). Returns steps, total reward, outcome, text trace."""
    obs, _ = env.reset(seed=episode_seed)
    trace = ["state %d\n" % obs + clean_render(env.render())]
    total, steps = 0.0, 0
    terminated = truncated = False
    for _ in range(max_steps):
        action = choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total += reward
        trace.append("step %d: action %s -> state %d, reward %g\n%s"
                     % (steps, ACTION_NAMES[action], obs, reward, clean_render(env.render())))
        if terminated or truncated:
            break
    if terminated:                       # a termination on the last allowed step still counts
        outcome = "goal" if obs == GOAL_STATE else "hole"
    else:
        outcome = "time_limit"
    return steps, total, outcome, "\n".join(trace)


def goal_probability(env, action_probs):
    """REFERENCE ONLY: exact chance of reaching the goal within MAX_STEPS steps, from the
    environment's own transition table. Used to judge the simulated batches, not to act."""
    try:
        P = env.unwrapped.P
        x = [0.0] * 16
        x[GOAL_STATE] = 1.0
        for _ in range(MAX_STEPS):
            new = x[:]
            for s in NONTERMINAL:
                new[s] = sum(pa * sum(p * x[s2] for p, s2, _, _ in P[s][a])
                             for a, pa in action_probs(s).items())
            x = new
        return x[0]
    except Exception:
        return None


def main():
    conditions = [(slip, pol) for slip in (False, True) for pol in ("random", "fixed")]
    episode_rows, batch_rows, cond_rows, notes = [], [], [], []
    example_written = False

    for slip, pol in conditions:
        name = "%s_%s" % ("slippery" if slip else "nonslippery", pol)
        env = make_env(slip)
        first_goal = first_fail = None
        for batch_seed in BATCH_SEEDS:
            rng = random.Random(batch_seed)          # seeded action generator
            if pol == "random":
                choose = lambda s, rng=rng: rng.randrange(4)
            else:
                choose = lambda s: FIXED_POLICY[s]
            counts = {"goal": 0, "hole": 0, "time_limit": 0}
            step_sum = 0
            for i in range(EPISODES_PER_BATCH):
                ep_seed = 1000 * batch_seed + i
                steps, total, outcome, trace = run_episode(env, choose, ep_seed)
                counts[outcome] += 1
                step_sum += steps
                episode_rows.append([name, "yes" if slip else "no", pol, batch_seed, ep_seed,
                                     steps, total, outcome])
                head = "%s, batch seed %d, episode seed %d, outcome %s, %d steps\n\n" % (
                    name, batch_seed, ep_seed, outcome, steps)
                if outcome == "goal" and first_goal is None:
                    first_goal = head + trace
                if outcome != "goal" and first_fail is None:
                    first_fail = head + trace
                if name == "nonslippery_fixed" and batch_seed == 11 and i == 0 and not example_written:
                    with open(os.path.join(OUT, "example_episode.txt"), "w", encoding="utf-8") as f:
                        f.write(head + trace)
                    example_written = True
            batch_rows.append(["yes" if slip else "no", pol, batch_seed, counts["goal"],
                               counts["hole"], counts["time_limit"],
                               100.0 * counts["goal"] / EPISODES_PER_BATCH,
                               step_sum / EPISODES_PER_BATCH])
        for label, text in (("success", first_goal), ("failure", first_fail)):
            if text is None:
                notes.append("%s: no %s episode was observed" % (name, label))
            else:
                with open(os.path.join(OUT, "traces", "%s_%s.txt" % (name, label)), "w",
                          encoding="utf-8") as f:
                    f.write(text)
        probs = (lambda s: {a: 0.25 for a in range(4)}) if pol == "random" \
            else (lambda s: {FIXED_POLICY[s]: 1.0})
        ref = goal_probability(env, probs)
        rows = [r for r in batch_rows if r[0] == ("yes" if slip else "no") and r[1] == pol]
        n = 3 * EPISODES_PER_BATCH
        cond_rows.append(["yes" if slip else "no", pol, sum(r[3] for r in rows),
                          sum(r[4] for r in rows), sum(r[5] for r in rows),
                          100.0 * sum(r[3] for r in rows) / n,
                          sum(r[7] for r in rows) / 3, "" if ref is None else 100.0 * ref])

    def write(name, header, rows):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("episodes.csv", ["condition", "slippery", "policy", "batch_seed", "episode_seed",
                           "steps", "total_reward", "outcome"], episode_rows)
    write("summary_by_batch.csv", ["slippery", "policy", "batch_seed", "goals_of_50", "holes_of_50",
                                   "time_limits_of_50", "success_percent", "mean_steps"], batch_rows)
    write("summary_by_condition.csv", ["slippery", "policy", "goals_of_150", "holes_of_150",
                                       "time_limits_of_150", "success_percent", "mean_steps",
                                       "reference_exact_goal_percent"], cond_rows)
    with open(os.path.join(OUT, "fixed_policy.txt"), "w", encoding="utf-8") as f:
        f.write("Fixed policy (action for every nonterminal state):\n")
        for s in NONTERMINAL:
            f.write("state %2d (row %d, col %d): %s\n" % (s, s // 4 + 1, s % 4 + 1, ACTION_NAMES[FIXED_POLICY[s]]))
    with open(os.path.join(OUT, "environment_info.txt"), "w", encoding="utf-8") as f:
        f.write("python %s\ngymnasium %s\nenv FrozenLake-v1 map_name=4x4 default rewards\n"
                % (platform.python_version(), gym.__version__))

    print("Gymnasium", gym.__version__)
    print("\n| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in batch_rows:
        print("| %s | %s | %d | %d | %d | %d | %.1f | %.2f |" % tuple(r))
    print("\nPer condition (150 episodes). Last column = exact chance of reaching the goal within"
          " 100 steps, reference only:")
    for r in cond_rows:
        print("  slippery=%s policy=%s goals=%d holes=%d time_limits=%d success=%.1f%% "
              "mean_steps=%.2f exact_reference=%s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                                                    "n/a" if r[7] == "" else "%.1f%%" % r[7]))
    for n in notes:
        print("NOTE:", n)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
