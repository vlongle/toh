import gym
import numpy as np
import cv2
from itertools import product
from gym import spaces
from copy import deepcopy
from typing import (
    Optional,
    Tuple,
    List,
)


class TowerOfHanoi(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    INVALID_MOVE_PENALTY = -1.0
    MOVE_COST = -0.01
    SUCCESS_REWARD = 1.0

    def __init__(self, num_discs: Optional[int] = 3,
                 allow_invalid_actions: Optional[bool] = True,
                 initial_state: Optional[np.ndarray] = None,
                 goal_state: Optional[np.ndarray] = None,
                 **kwargs) -> None:
        """
        Args:
            num_discs: 
                Number of discs in the game
            allow_invalid_actions: 
                Whether to allow invalid actions. If False, we use action masks 
                to prevent invalid actions.
            initial_state: 
                Initial state of the game. If None, we use the default initial state.
            goal_state: 
                Goal state of the game. If None, we use the default goal state.
        """
        self.num_discs = num_discs
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(num_discs,), dtype=np.int32)  # e.g., [0, 0] means all two discs are on
        # the leftmost peg

        self.action_space = spaces.MultiDiscrete(
            [3, 3])  # (from_tower, to_tower) tuple

        self.allow_invalid_actions = allow_invalid_actions

        # user can override default INITIAL_STATE that the env resets to
        # all discs on leftmost peg
        self.INITIAL_STATE = np.zeros(num_discs, dtype=np.int32)
        if initial_state is not None:
            self.INITIAL_STATE = initial_state

        # user can also override the default goal state
        # all discs on rightmost peg
        self.GOAL_STATE = np.ones(num_discs, dtype=np.int32) * 2
        if goal_state is not None:
            self.GOAL_STATE = goal_state

        self._state = None  # np.ndarray of shape (num_discs,)
        # list of length 3, where towers[i] is a list of disc sizes on tower i
        # this is a convenience view of self._state for game logic, and is kept in sync with self._state
        self._towers = None

        # change the class attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def state(self) -> np.ndarray:
        return None if self._state is None else np.copy(self._state)

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = np.copy(value)
        # keep towers in-sync with state
        self._towers = [np.where(self._state == i)[0].tolist()
                        for i in range(3)]

    @property
    def towers(self) -> List:
        return deepcopy(self._towers)

    @classmethod
    def is_valid_move_(self, state: np.ndarray, action: np.ndarray) -> bool:
        from_tower, to_tower = action
        if state[state == from_tower].size == 0:
            return False  # from_tower is empty
        if state[state == to_tower].size == 0:
            return True  # to_tower is empty
        # move_disc is the argmin of the indices of the discs on from_tower
        move_disc = min(np.where(state == from_tower)[0])
        target_disc = min(np.where(state == to_tower)[0])
        return move_disc < target_disc

    def is_valid_move(self, action: np.ndarray) -> bool:
        return self.is_valid_move_(self.state, action)

    @classmethod
    def move(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        assert self.is_valid_move_(state, action), "Invalid move"
        from_tower, to_tower = action
        move_disc = min(np.where(state == from_tower)[0])
        state[move_disc] = to_tower
        return state

    # def action_masks(self) -> np.ndarray:
    #     possible_actions = product(range(3), range(3))
    #     print(possible_actions)
    #     masks = np.ones(len(possible_actions), dtype=np.int32)
    #     for action in possible_actions:
    #         if not self.is_valid_move(action):
    #             masks[action] = 0
    #     return masks

    def reset(self) -> np.ndarray:
        self.state = self.INITIAL_STATE
        return self.state

    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, dict]:
        assert self.state is not None,  "Cannot call env.step() before calling reset()"
        is_valid_move = self.is_valid_move(action)
        if is_valid_move:
            self.state = self.move(self.state, action)
            reward = self.MOVE_COST
        else:
            reward = self.INVALID_MOVE_PENALTY
        done = np.array_equal(self.state, self.GOAL_STATE)
        reward += done * self.SUCCESS_REWARD
        obs = self.state
        info = {"is_success": done,
                "is_valid_move": is_valid_move}
        return obs, reward, done, info

    def render(self, mode="rgb_array") -> np.ndarray:
        blue = (78, 162, 196)
        green = (77, 206, 145)
        grey = (170, 170, 170)
        SCREEN_WIDTH = 800
        # SCREEN_HEIGHT = 300 + DISC_HEIGHT * self.num_discs
        SCREEN_HEIGHT = 300
        DISC_HEIGHT = 20
        DISC_MAX_WIDTH = 100  # width of the bases of the towers
        DISC_HEIGHT_SEP = 5  # vertical separation between discs
        # 0.8 means that the width of the second largest disc is 80% of the width of the largest disc
        DISC_WIDTH_FACTOR = 0.8
        SPACE_BETWEEN_TOWERS = 100
        PEG_WIDTH = 10
        TOWER_HEIGHT = (DISC_HEIGHT + DISC_HEIGHT_SEP)

        # initialize white screen
        screen = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                         dtype=np.uint8) * 255

        def draw_towers():
            tower_width = DISC_MAX_WIDTH
            # tower at the bottom of the screen separated by SPACE_BETWEEN_TOWERS
            for i in range(3):
                x = (i + 1) * SPACE_BETWEEN_TOWERS + i * tower_width
                y = SCREEN_HEIGHT - TOWER_HEIGHT
                cv2.rectangle(screen, (x, y), (x + tower_width,
                              y + TOWER_HEIGHT), grey, -1)

        def draw_pegs():
            peg_height = (self.num_discs + 1) * (DISC_HEIGHT + DISC_HEIGHT_SEP)
            for i in range(3):
                x = (i + 1) * SPACE_BETWEEN_TOWERS + i * \
                    DISC_MAX_WIDTH + DISC_MAX_WIDTH // 2
                y = SCREEN_HEIGHT - TOWER_HEIGHT - peg_height
                cv2.rectangle(screen, (x, y), (x + PEG_WIDTH,
                              y + peg_height), green, -1)

        def draw_discs():
            tower_width = DISC_MAX_WIDTH
            # largest disc is at the bottom of the tower and the smallest disc is at the top of the tower/peg
            for i in range(3):
                x = (i + 1) * SPACE_BETWEEN_TOWERS + i * tower_width
                y = SCREEN_HEIGHT - TOWER_HEIGHT - \
                    (DISC_HEIGHT + DISC_HEIGHT_SEP) * self.num_discs
                for j, disc in enumerate(sorted(self.towers[i])):
                    disc_width = int(
                        tower_width * DISC_WIDTH_FACTOR ** (self.num_discs - disc))
                    disc_x = x + (tower_width - disc_width) // 2
                    disc_y = y + j * (DISC_HEIGHT + DISC_HEIGHT_SEP)
                    cv2.rectangle(screen, (disc_x, disc_y), (disc_x +
                                  disc_width, disc_y + DISC_HEIGHT), blue, -1)
        draw_towers()
        draw_pegs()
        draw_discs()
        return screen

    @classmethod
    def minimum_num_moves(cls, num_discs) -> int:
        """
        Returns the minimum number of moves required to solve the game
        i.e. the number of moves required to move all discs from the first tower to the third tower
        More info: https://en.wikipedia.org/wiki/Tower_of_Hanoi
        """
        return 2 ** num_discs - 1

    @classmethod
    def tower_of_hanoi_solver(cls, n_discs) -> List[Tuple[int, int]]:
        # generate the trajectory that solves the tower of hanoi
        # with n_discs using dynamic programming
        # https://en.wikipedia.org/wiki/Tower_of_Hanoi#Recursive_solution
        def move(n, source, target, aux):
            if n > 0:
                move(n - 1, source, aux, target)
                moves.append((source, target))
                move(n - 1, aux, target, source)
        moves = []
        move(n_discs, 0, 2, 1)
        return moves

    @classmethod
    def build_tree(cls, num_discs):
        """
        use bfs to build the tree of all possible states
        in the tower of hanoi game with num_discs
        """
        from collections import deque
        queue = deque()
        root = np.array([0] * num_discs)
        queue.append(root)
        possible_actions = list(product(range(3), range(3)))
        possible_states = list(product(range(3), repeat=num_discs))
        adjacencies = {state: [] for state in possible_states}

        def add_adjacency(state1, state2):
            """
            bidirectional graph since every action in the game
            is reversible
            """
            state1 = tuple(state1)
            state2 = tuple(state2)
            print("adding adjacency", state1, state2)
            adjacencies[state1].append(state2)
            # adjacencies[state2].append(state1)

        visited = set()
        while queue:
            level_size = len(queue)
            print(f"level size: {level_size}")
            for _ in range(level_size):
                state = queue.popleft()
                print("state: ", state)
                visited.add(tuple(state))
                print("possible actions: ", possible_actions)
                for action in possible_actions:
                    print("action: ", action, "is_valid: ",
                          cls.is_valid_move_(state, action))
                    if cls.is_valid_move_(state, action):
                        new_state = cls.move(np.copy(state), action)
                        add_adjacency(state, new_state)
                        if tuple(new_state) not in visited:
                            queue.append(new_state)
        return adjacencies
    # @classmethod
    # def compute_V_value(cls, n_discs, goal_state) -> float:
    #     possible_states = list(product(range(3), repeat=n_discs))
    #     # solve the linear system of equations
    #     # given by the Bellman equation
    #     # V(s | pi) = R(s, s') + gamma * V(s' | pi)
    #     gamma = 1.0
    #     A = np.zeros((len(possible_states), len(possible_states)))
    #     b = np.zeros(len(possible_states))
    #     for i, state in enumerate(possible_states):
    #         if state == goal_state:
    #             b[i] = cls.SUCCESS_REWARD
    #             A[i, i] = 1
    #         else:
    #             b[i] = cls.MOVE_COST
    #             A[i, i] = 1
    #             for j, next_state in enumerate(possible_states):
    #                 if cls.is_valid_move(state, next_state):
    #                     A[i, j] = -gamma
    #     V = np.linalg.solve(A, b)
    #     return V[0]
