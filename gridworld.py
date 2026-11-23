"""Grid World model. The floor rules live in data/maps.json, NOT in the solver.

To add a new floor type, add a tile to maps.json (reward, movement, ...) and use its
letter in a map. Nothing in this file or in value_iteration.py has to change.
Coordinates are (row, column) and start at 1, as in the assignment.
"""
import json

DIRECTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}
ACTIONS = ("up", "right", "down", "left")        # order also used to break ties


def load_config(path="data/maps.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class GridWorld:
    def __init__(self, rows, tiles, discount, terminal_value=0.0):
        self.rows = list(rows)
        self.tiles = tiles
        self.discount = discount
        self.terminal_value = terminal_value
        self.n_rows, self.n_cols = len(self.rows), len(self.rows[0])
        self.grid = {}
        for r, line in enumerate(self.rows, start=1):
            assert len(line) == self.n_cols, "all map rows must have the same length"
            for c, ch in enumerate(line, start=1):
                assert ch in tiles, "unknown tile letter %r (add it to maps.json)" % ch
                self.grid[(r, c)] = ch

    # ---- tiles ---------------------------------------------------------------
    def letter(self, pos):
        return self.grid[pos]

    def is_wall(self, pos):
        return self.tiles[self.grid[pos]].get("traversable", True) is False

    def is_terminal(self, pos):
        return bool(self.tiles[self.grid[pos]].get("terminal", False))

    def reward_of(self, pos):
        return float(self.tiles[self.grid[pos]]["reward"])

    def states(self):
        return [p for p in self.grid if not self.is_wall(p)]

    def nonterminal_states(self):
        return [p for p in self.states() if not self.is_terminal(p)]

    def start(self):
        return next(p for p, ch in self.grid.items() if ch == "S")

    # ---- dynamics ------------------------------------------------------------
    def move(self, pos, direction):
        """Where a move in `direction` ends: a wall or the map edge leaves you in place."""
        dr, dc = DIRECTIONS[direction]
        new = (pos[0] + dr, pos[1] + dc)
        if new not in self.grid or self.is_wall(new):
            return pos
        return new

    def direction_probabilities(self, pos, action):
        """(probability, actual direction) pairs. Slipping depends on the tile you LEAVE."""
        rule = self.tiles[self.grid[pos]]
        if rule.get("movement") == "slippery":
            intended = rule.get("intended", 0.8)
            side = round((1.0 - intended) / 2.0, 12)   # 0.1, without float noise
            left_side, right_side = PERPENDICULAR[action]
            return [(intended, action), (side, left_side), (side, right_side)]
        return [(1.0, action)]

    def transitions(self, pos, action):
        """List of (probability, next position, reward, direction). Empty for terminals:
        nothing happens, and no reward is paid, after termination."""
        if self.is_terminal(pos):
            return []
        out = []
        for prob, direction in self.direction_probabilities(pos, action):
            nxt = self.move(pos, direction)
            out.append((prob, nxt, self.reward_of(nxt), direction))   # reward of destination
        return out


def build_world(config, map_key):
    return GridWorld(config[map_key], config["tiles"], config.get("discount", 0.9),
                     config.get("terminal_value", 0))
