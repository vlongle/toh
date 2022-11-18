from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from tower_of_hanoi import TowerOfHanoi
import unittest
import numpy as np
import warnings


env = TowerOfHanoi(num_discs=3, SUCCESS_REWARD=2.0)
assert env.SUCCESS_REWARD == 2.0, "Env should support changing class attributes"


class TestTowerOfHanoi(unittest.TestCase):

    def test_default_class_attributes(self):
        env = TowerOfHanoi(num_discs=3)
        assert env.SUCCESS_REWARD == 1.0, "Default SUCCESS_REWARD should be 1.0"
        assert env.MOVE_COST == -0.01, "Default MOVE_COST should be -0.01"
        assert env.INVALID_MOVE_PENALTY == - \
            1.0, "Default INVALID_MOVE_PENALTY should be -1.0"

    def test_changing_class_attributes(self):
        env = TowerOfHanoi(num_discs=3, SUCCESS_REWARD=2.0)
        assert env.SUCCESS_REWARD == 2.0, "Env should support changing class attributes"

    def test_reset(self):
        env = TowerOfHanoi(num_discs=3)
        assert env.state is None, "Env state should be None before reset"
        assert env.towers is None, "Env towers should be None before reset"
        env.reset()
        assert np.array_equal(
            env.reset(),
            env.INITIAL_STATE),\
            "Env should reset to default INITIAL_STATE"
        env = TowerOfHanoi(num_discs=3, initial_state=[1, 1, 1])
        assert np.array_equal(env.reset(),
                              np.array([1, 1, 1]),
                              "Env should reset to user-specified INITIAL_STATE")

    def test_towers(self):
        env = TowerOfHanoi(num_discs=3)
        env.reset()
        assert env.towers == [[0, 1, 2], [], []
                              ], "Towers should be in sync with state"
        # change state should change towers
        env.state = np.array([1, 1, 1])
        assert env.towers == [[], [0, 1, 2], []
                              ], "Towers should be in sync with state"

    def test_step(self):
        env = TowerOfHanoi(num_discs=3)
        with self.assertRaises(AssertionError, msg="step() should raise error since reset() has not been called"):
            env.step([0, 1])
        env.reset()
        assert np.array_equal(
            env.state, env.INITIAL_STATE), "Env should reset to default INITIAL_STATE"
        assert env.towers == [[0, 1, 2], [], []
                              ], "Towers should be in sync with state"

        # ====== Test valid moves ======
        obs, reward, done, info = env.step([0, 1])
        assert np.array_equal(
            env.state, np.array([1, 0, 0])), "State should be updated after step()"
        assert np.array_equal(
            obs, env.state), "Obs should be the same as state"
        assert env.towers == [[1, 2], [0], []
                              ], "Towers should be in sync with state"
        assert np.array_equal(
            reward, env.MOVE_COST), "Reward should be MOVE_COST"
        assert done is False, "Done should be False"
        assert info == {"is_success": False,
                        "is_valid_move": True}, "Info after valid move should be correct"

        # ==== Test invalid moves ====
        obs, reward, done, info = env.step([0, 1])
        assert np.array_equal(
            env.state, np.array([1, 0, 0])), "State should not be updated after invalid step()"
        assert np.array_equal(
            obs, env.state), "Obs should be the same as state"
        assert env.towers == [[1, 2], [0], []
                              ], "Towers should be in sync with state"
        assert np.array_equal(
            reward, env.INVALID_MOVE_PENALTY), "Reward should be INVALID_MOVE_PENALTY"
        assert done is False, "Done should be False"
        assert info == {"is_success": False,
                        "is_valid_move": False}, "Info after invalid move should be correct"

        # ==== Test success ====
        env.state = np.array([0, 2, 2])
        obs, reward, done, info = env.step([0, 2])
        assert np.array_equal(
            env.state, np.array([2, 2, 2])), "State should be updated after step()"
        assert np.array_equal(
            obs, env.state), "Obs should be the same as state"
        assert np.array_equal(
            list(map(set, env.towers)), [set(), set(), set({0, 1, 2})]), "Towers should be in sync with state"
        assert np.array_equal(
            reward, env.SUCCESS_REWARD + env.MOVE_COST), "Reward should be SUCCESS_REWARD + MOVE_COST"
        assert done is True, "Done should be True"
        assert info == {"is_success": True,
                        "is_valid_move": True}, "Info after success should be correct"

    # def test_action_masks(self):
    #     pass


if __name__ == '__main__':
    # unittest.main()
    env = TowerOfHanoi(num_discs=2)
    from pprint import pprint
    pprint(env.build_tree(num_discs=2))
    # TIME_LIMIT = 50
    # NUM_EPISODES = 200
    # solver = PPO("MlpPolicy", env, verbose=1,  device="cuda",
    #              n_steps=min(2048, TIME_LIMIT),
    #              seed=0,
    #              )
    # solver.learn(total_timesteps=TIME_LIMIT *
    #              NUM_EPISODES, progress_bar=True,
    #              log_interval=20)
    # print(">> Evaluation:", evaluate_policy(solver, env, n_eval_episodes=1,
    #                                         deterministic=True, render=False,
    #                                         warn=False, return_episode_rewards=True))


"""
    0 1 2
0  [      ] [V(0)]
1  [      ] [V(1)]
2  [      ] [V(2)]

"""
