HW6 submission

Student: [TODO: your name and group]

AI-use declaration

I used Claude (Anthropic's AI assistant) during this homework. It was used for:

explaining the ideas of the assignment (MDPs, value iteration, how FrozenLake and Gymnasium work);
worked solutions and explanations for the paper calculations in Part A, which I then went through and copied by hand onto paper;
designing the fixed FrozenLake policy with me, and writing the Python code (run_part_b.py, gridworld.py, value_iteration.py, run_part_c.py) and a first draft of this file;
explaining the program output.

I ran all programs myself on my own computer, and the numbers in this file come from my own runs. [TODO: check that this matches your course's AI rules (HW4) and edit it so it is exactly true for you.]

Setup and reproduction
Python 3.12, Windows, virtual environment .venv. [TODO: check with python --version]
Gymnasium 1.4.0 (printed by the program). Other versions are in requirements.txt. [TODO: create it with pip freeze > requirements.txt and, if you like, add the numpy version here]
Only Gymnasium is used for Part B (FrozenLake-v1, map_name="4x4", default rewards and slip probabilities). Part C uses no library: the solver and the simulator are my own code.

Commands (from the project folder):

bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Mac/Linux: source .venv/bin/activate)
pip install gymnasium numpy

python run_part_b.py              # Part B -> outputs/part_b/
python run_part_c.py              # Part C -> outputs/part_c/

Files: run_part_b.py (Part B). gridworld.py (the Grid World model; all floor rules are read from data/maps.json, so a new floor needs no solver change), value_iteration.py (value iteration and policy extraction), run_part_c.py (checks, solving both maps, simulation, comparison).

Part A — Paper calculations

[TODO: link your scan/photo, or write "Shown to the instructor in person on DATE". Note that the README asks to commit the Part A photo before the first program; if you did not, say so honestly.]

Summary of the paper results (all match the program, see Part C):

MDP: states are the nonwall tiles (terminal G and H have value 0); actions are up/right/down/left; the reward is that of the tile where the move ends; on ice the move goes as chosen with probability 0.8 and sideways with 0.1 each. The agent maximizes the expected discounted sum of rewards with discount 0.9.
V1: −1 on every nonterminal tile except (1,3) = 10.
 -1 | -1 | 10 |  G
 -1 |  # | -1 |  H
 -1 | -1 | -1 | -1
V2 from V1: (1,2) = 8.0, best action right (up −3.9, right 8.0, down −3.9, left −1.9). (1,3) = 10, best action right (up 8.0, right 10, down −1.9, left −3.9). (2,3) ice = 5.21, best action up (up 5.21, right −7.39, down −2.71, left −0.91). No ties at these three positions.
Why two rounds do not prove the final policy: V2 looks only two steps ahead. Tiles far from the goal have not yet "seen" it, and values and best actions can still change in later rounds. A policy should be read from values that have stopped changing.
Part B — FrozenLake
Fixed policy, written before the experiments

The policy was fixed in the code before any experiment was run and was not changed afterwards. It follows the shortest route that never steps on a hole: 0 ↓ 4 ↓ 8 → 9 → 10 ↓ 14 → 15. State numbers are row·4 + column (rows and columns from 0), and the map is SFFF / FHFH / FFFH / HFFG. Holes are states 5, 7, 11, 12 and the goal is 15, so there are 11 nonterminal states. Full list in fixed_policy.txt.

State	Action	Reasoning
0	Down	start the route along the left edge
1	Right	down would enter the hole at 5
2	Down	toward the bottom
3	Left	down would enter the hole at 7
4	Down	right would enter the hole at 5
6	Down	toward the bottom row
8	Right	down would enter the hole at 12
9	Right	toward 10
10	Down	toward 14
13	Right	toward 14
14	Right	into the goal 15
Results

Setup: 3 batches of 50 episodes per condition, batch seeds 11, 22, 33; episode i is reset with seed 1000 × batch_seed + i; the random action generator is seeded with the batch seed; the step limit is 100. The same episode seeds are used in all four conditions.

Slippery?	Policy	Batch seed	Goals /50	Holes /50	Time limits /50	Success %	Mean steps
no	random	11	2	48	0	4.0	9.78
no	random	22	0	50	0	0.0	6.90
no	random	33	0	50	0	0.0	7.44
no	fixed	11	50	0	0	100.0	6.00
no	fixed	22	50	0	0	100.0	6.00
no	fixed	33	50	0	0	100.0	6.00
yes	random	11	0	50	0	0.0	8.44
yes	random	22	1	49	0	2.0	7.76
yes	random	33	0	50	0	0.0	7.30
yes	fixed	11	3	47	0	6.0	4.52
yes	fixed	22	2	48	0	4.0	5.06
yes	fixed	33	0	50	0	0.0	4.06

Over all 150 episodes per condition:

Slippery?	Policy	Goals	Holes	Time limits	Success %	Mean steps	Exact chance of the goal within 100 steps (reference)
no	random	2	148	0	1.3	8.04	1.4 %
no	fixed	150	0	0	100.0	6.00	100.0 %
yes	random	1	149	0	0.7	7.83	1.4 %
yes	fixed	5	145	0	3.3	4.55	3.7 %

The last column is computed from the environment's own transition table. It is used only to judge my simulated batches, never by the policy.

Files:

one row per episode: episodes.csv; batch and condition summaries: summary_by_batch.csv, summary_by_condition.csv; versions: environment_info.txt
one episode with text rendering: example_episode.txt
traces in outputs/part_b/traces/: a success and a failure for the slippery fixed policy (slippery_fixed_success.txt, slippery_fixed_failure.txt), the same for the random policy in both modes, and a success for the non-slippery fixed policy (nonslippery_fixed_success.txt).
Not observed: the fixed policy never failed without slipping (150 of 150 reached the goal), so there is no failed trace for that condition. No episode ever hit the 100-step limit, so there is no time-limit trace either.

What slipping changed. Without slipping the fixed policy is deterministic: it reached the goal in all 150 episodes in exactly 6 steps. With slipping it succeeded in only 5 of 150 (3.3 %, exact chance 3.7 %). Each move can turn sideways, and many of the route's tiles are next to holes, so most episodes end in a hole after about 4–5 steps (the mean steps fell from 6.00 to 4.55 because falling into a hole ends the episode early). The random policy was not changed by slipping: 1.3 % without and 0.7 % with slipping, while the exact chance is 1.4 % in both cases. This is expected: a uniformly random action followed by a sideways slip is still a uniformly random direction, so slipping cannot change the random policy's behaviour. The difference between 1.3 % and 0.7 % is only noise (2 goals against 1).

Variation across batches. The batches of the same condition differ: the slippery fixed policy gave 6 %, 4 % and 0 %. With an exact chance of 3.7 %, a batch of 50 is expected to contain about 1.9 goals with a standard deviation of about 1.3, so 0 to 3 goals is normal. The non-slippery fixed policy gave exactly the same result in all batches (100 %, 6 steps), because there is no randomness in the environment to vary.

Why one successful episode is not enough. Even the weak policies reach the goal sometimes: the random policy won once in the slippery condition, and the slippery fixed policy wins about 4 times in 100. One win is therefore easy to get by luck and says very little about the policy. Only many episodes, and a comparison of the batches, show how often a policy really succeeds.

Why these runs are not training. Nothing in these runs learns: the policies never change, there are no value estimates and no updates, and the seeds only make the runs repeatable. I only measured how often each fixed policy reaches the goal. Each policy behaved in the last episode exactly as in the first one.

Part C — Custom solver
Compare code with paper

The program reproduces my paper results exactly (largest difference 0.00):

V1 = −1 on every nonterminal tile except (1,3) = 10.
  -1.0000    -1.0000    10.0000     G
  -1.0000     #         -1.0000     H
  -1.0000    -1.0000    -1.0000    -1.0000
V2 from V1:
Position	up	right	down	left	V2	Best action	Paper
(1,2) mud	−3.9000	8.0000	−3.9000	−1.9000	8.0000	right	8.0000
(1,3)	8.0000	10.0000	−1.9000	−3.9000	10.0000	right	10.0000
(2,3) ice	5.2100	−7.3900	−2.7100	−0.9100	5.2100	up	5.2100

There were no discrepancies to explain. [TODO: if your own paper numbers differed anywhere, describe it here and show your correction.]

Values, policies and convergence

Both maps: paper_map_values_policy.csv and extended_map_values_policy.csv (value, chosen action, tied actions and the four action values for every tile), iteration histories in paper_map_iterations.csv and extended_map_iterations.csv, the full printout in run_log.txt.

Map	Iterations	Final largest change	Converged
Part A map	12	2.165 · 10⁻⁷	yes (tolerance 0.000001, safeguard 10,000)
Extended map	11	2.406 · 10⁻⁷	yes

Tie-breaking rule: actions whose value is within 10⁻⁹ of the best are counted as tied, and the first of the order up, right, down, left is chosen. No ties occurred in either final policy.

Part A map (final values, then policy):

 4.2000   8.0000  10.0000    G          >  >  >  G
 2.7800      #     5.8242    H          ^  #  ^  H
 1.5358   2.8176   4.2418   2.8176      >  >  ^  <

Extended map .M.G / .#IH / SD.W:

 4.2000   8.0000  10.0000    G          >  >  >  G
 2.7800      #     5.8242    H          ^  #  ^  H
 1.5020   2.8176   4.2418   2.8176      ^  >  ^  <
Simulation results

The computed policy was simulated with the Part B seed scheme (batch seeds 11, 22, 33; episode seed 1000 × batch_seed + i), 50 episodes per batch, step limit 100. The discounted return is r0 + 0.9·r1 + 0.9²·r2 + ….

Map	Batch seed	Goals /50	Holes /50	Time limits /50	Mean steps	Mean discounted return
Part A map	11	42	8	0	4.98	0.8913
Part A map	22	46	4	0	5.04	1.9148
Part A map	33	48	2	0	5.04	2.4921
Extended map	11	50	0	0	5.00	1.5020
Extended map	22	50	0	0	5.00	1.5020
Extended map	33	50	0	0	5.00	1.5020

Episode rows: paper_map_simulation_episodes.csv, extended_map_simulation_episodes.csv; batch rows: paper_map_simulation_batches.csv, extended_map_simulation_batches.csv (all in outputs/part_c/).

Explain your results

Which values and actions changed with sand and water? Only the start tile changed: its action went from right to up, and its value fell from 1.5358 to 1.5020. All other values and actions stayed the same (comparison.csv). The reward is paid for entering a tile, so the sand tile (3,2) and the water tile (3,4) do not change their own values. The new floor costs show up only for tiles that enter them. The old best route from the start went right through (3,2). With sand costing −2 instead of −1, that route is worth 0.5358 from the start (−2 + 0.9 · 2.8176), against 1.5020 for going up (−1 + 0.9 · 2.78), so the start now goes up. The optimal policy never needs the water tile, so its cost changes nothing.

Where did cost and risk matter? The cost mattered at the start: sand made the right-hand route 1.0 worse, so the policy took the mud route instead. The risk mattered at the ice tile (2,3). The policy goes up there in both maps (value 5.8242). The alternatives are worse: left 4.4752 (it bumps into the wall and stays on ice), down 1.6782, right −6.9182 (the hole is the intended direction). Going up accepts a 10 % chance of sliding into the hole at (2,4). In the Part A map the start tile uses the icy route, because it is slightly better (1.5358 against 1.5020 for the mud route). In the extended map the start tile avoids the ice completely.

How do simulations relate to the computed values? The value V*(S) is the expected discounted return from the start under the optimal policy, over an unlimited number of steps. A simulation is a finite sample of this expectation, so it is close to the value but not identical. On the extended map the policy never touches the ice and every move is deterministic, so each of the 150 episodes follows the same route and has exactly the return 1.5020 = V*(S). On the Part A map the policy uses the ice, so the episodes differ: 121 of the 150 reached the goal in 5 steps with return 3.122, 15 slid sideways into the wall, stayed one extra step on the ice and then reached the goal with return 1.81, and 14 fell into the hole with −10.0. Because the returns are so spread out (standard deviation about 3.8), the batch means are far apart (0.89, 1.91, 2.49) and the mean over all 150 episodes is 1.77. This is 0.23 above V*(S) = 1.54, less than one standard error (about 0.31), so it agrees with the value within the sampling noise. The 100-step limit has a negligible effect because 0.9¹⁰⁰ ≈ 3 · 10⁻⁵, and no episode reached the limit.

Rule checks

All 23 checks passed before solving; they are printed at the top of run_log.txt with the transitions they inspected.

Check	Result
Map edge: moving up from (1,1)	stays at (1,1), reward −1 (the tile it stays on)
Map edge: moving right from (3,4)	stays at (3,4), reward −1
Wall: moving right from (2,1) into the wall (2,2)	stays at (2,1), reward −1
Wall on mud: moving down from (1,2)	stays at (1,2), reward −3
Ice (2,3), all four actions	three outcomes (0.8, 0.1, 0.1), probabilities sum to 1
Ice, action up	0.8 to (1,3), 0.1 blocked by the wall (stays), 0.1 to the hole (2,4)
Slipping depends on the tile left	moving onto the ice from (1,3) is deterministic
Mud reward	entering (1,2) gives −3
Sand reward (extended map)	entering (3,2) gives −2
Water reward (extended map)	entering (3,4) gives −5
Sand and water	deterministic (a single outcome)
Goal and hole	entering G gives +10, entering H gives −10
After termination	G and H have no transitions in either map, so no rewards after the end
Wall is not a state	(2,2) is not in the list of states
Rollout from (1,3), action right	total reward exactly +10 in 1 step