"""Reward regressions: conveyor objects that leave the track are removed and never count; the count box rewards
honest partial counts in equal steps and gives nothing for over-counting or off-directive objects."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from abc_sim import make_env


def _advance(env, seconds: float) -> None:
    env.advance_physics(round(seconds / env.model.opt.timestep))


def _free_qpos_adr(model: mujoco.MjModel, joint: str) -> int:
    return int(
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    )


class ConveyorOffTrackTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("conveyor_pick", render_cameras=False)
        self.env.reset(seed=3)

    def test_objects_riding_off_the_belt_are_removed(self):
        # Idle arms: every object reaches the belt's end within a few seconds and drops onto the table.
        _advance(self.env, 12.0)
        result = self.env.evaluate_task()
        self.assertEqual(int(result.metrics["num_removed_objects"][0]), 3)
        self.assertEqual(int(result.metrics["num_objects_in_bin"][0]), 0)
        self.assertTrue(bool(result.metrics["ever_failed"][0]))
        self.assertEqual(float(result.reward[0]), 0.0)
        self.assertEqual(
            len(self.env._task_runtime.debug_state()["removed_objects"]), 3
        )

    def test_an_object_in_the_bin_still_counts(self):
        evaluator = self.env._task_evaluator
        adr = _free_qpos_adr(self.env.model, "conveyor_object_1_joint")
        x, y = np.mean(evaluator._bin_x_bounds), np.mean(evaluator._bin_y_bounds)
        self.env.data.qpos[adr : adr + 3] = (x, y, evaluator._bin_floor_z + 0.08)
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        _advance(self.env, 12.0)
        result = self.env.evaluate_task()
        self.assertEqual(result.metrics["objects_in_bin"][0], ["conveyor_object_1"])
        self.assertEqual(int(result.metrics["num_removed_objects"][0]), 2)
        self.assertAlmostEqual(float(result.reward[0]), 1.0 / 3.0, places=6)
        self.assertFalse(bool(result.success[0]))


class CountBoxRewardTest(unittest.TestCase):
    @staticmethod
    def _reward(env, names):
        """Reward and success with ``names`` placed inside the box (positions only, no physics)."""
        evaluator = env._task_evaluator
        qpos = env.data.qpos.copy()
        x, y = np.mean(evaluator._box_x_bounds), np.mean(evaluator._box_y_bounds)
        for name in names:
            adr = _free_qpos_adr(env.model, name + "_joint")
            qpos[adr : adr + 3] = (x, y, evaluator._box_floor_z + 0.02)
        result = evaluator.evaluate_qpos_batch(qpos[None])
        return float(result.reward[0]), bool(result.success[0])

    def test_exact_count_rewards_each_object_up_to_the_target_and_nothing_past_it(self):
        env = make_env("count_three_into_opaque_box", render_cameras=False)
        env.reset(seed=0)
        names = env._task_evaluator._directive["names"]
        rewards = [self._reward(env, names[:k]) for k in range(6)]
        self.assertEqual(
            [success for _, success in rewards],
            [False, False, False, True, False, False],
        )
        np.testing.assert_allclose(
            [reward for reward, _ in rewards],
            [0.0, 1 / 3, 2 / 3, 1.0, 0.0, 0.0],
            rtol=1e-6,
        )

    def test_an_off_directive_object_zeroes_the_reward(self):
        env = make_env("count_into_opaque_box", render_cameras=False)
        for seed in range(50):
            env.reset(seed=seed)
            directive = env._task_evaluator._directive
            if not directive["eligible_mask"].all() and directive["target_count"] >= 2:
                break
        else:
            self.fail(
                "no sampled directive with ineligible objects and a target of at least 2"
            )
        names = np.asarray(directive["names"])
        eligible = list(names[directive["eligible_mask"]])
        ineligible = list(names[~directive["eligible_mask"]])
        target = directive["target_count"]
        self.assertAlmostEqual(
            self._reward(env, eligible[:1])[0], 1.0 / target, places=6
        )
        self.assertEqual(self._reward(env, eligible[:1] + ineligible[:1]), (0.0, False))
        self.assertEqual(
            self._reward(env, eligible[:target] + ineligible[:1]), (0.0, False)
        )
        self.assertEqual(self._reward(env, eligible[:target]), (1.0, True))


if __name__ == "__main__":
    unittest.main()
