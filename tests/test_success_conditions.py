"""CPU regressions for the revised simulator success conditions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from abc_sim.config import RobotSystemConfig
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.tasks.ball_tray import BallTrayBalancingRandomizer
from abc_sim.task_eval.ball_tray import (
    BallTrayBalancingEvaluator,
    sample_hold_duration_s,
)
from abc_sim.task_eval.chess import ChessBoardSetupEvaluator
from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_specs import get_task_spec


def body(name, joint, pos, extra=""):
    return (
        f'<body name="{name}" pos="{pos}"><freejoint name="{joint}"/>'
        f'<geom type="sphere" size=".01" mass=".1"/>{extra}</body>'
    )


def position(model, data, joint, xyz):
    address = int(model.jnt_qposadr[model.joint(joint).id])
    data.qpos[address : address + 3] = xyz


class BallTrayTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="table_plane" type="plane" size="1 1 .1"/>'
            '<body name="ball_tray" pos="0 0 .009"><freejoint name="ball_tray_joint"/>'
            '<geom type="box" size=".15 .15 .01" mass="1"/>'
            '<site name="ball_tray_center_site"/></body>'
            + body("ball_tray_ball", "ball_tray_ball_joint", "0 0 .039")
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.randomization = SimpleNamespace(seed=12, metadata={"hold_duration_s": 5.0})
        self.evaluator = BallTrayBalancingEvaluator(
            model=self.model, spec=get_task_spec("ball_tray_balancing")
        )
        self.evaluator.configure_from_randomization(self.model, self.randomization)
        self.sample(0)

    def sample(self, timestamp, *, lifted=False, ball_inside=True, contact=True):
        z = 0.109 if lifted else (0.009 if contact else 0.025)
        position(self.model, self.data, "ball_tray_joint", [0, 0, z])
        position(
            self.model,
            self.data,
            "ball_tray_ball_joint",
            [0 if ball_inside else 1, 0, z + 0.03],
        )
        self.data.time = timestamp
        mujoco.mj_forward(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def test_hold_then_contact_ignores_ball_after_hold(self):
        self.sample(1, lifted=True)
        self.assertFalse(self.sample(5.99, lifted=True).scalar_success())
        reached = self.sample(6, lifted=True)
        self.assertTrue(reached.metrics["hold_complete"][0])
        self.assertFalse(reached.scalar_success())
        self.assertFalse(
            self.sample(7, ball_inside=False, contact=False).scalar_success()
        )
        self.assertTrue(self.sample(8, ball_inside=False).scalar_success())
        self.assertTrue(self.sample(9, lifted=True, ball_inside=False).scalar_success())

    def test_interrupted_hold_restarts_and_can_retry(self):
        self.sample(1, lifted=True)
        self.sample(4, lifted=True, ball_inside=False)
        self.sample(5, lifted=True)
        self.assertFalse(self.sample(9.99, lifted=True).metrics["hold_complete"][0])
        self.assertTrue(self.sample(10, lifted=True).metrics["hold_complete"][0])
        self.assertTrue(self.sample(11).scalar_success())

    def test_time_is_required_for_qpos_only_calls(self):
        with self.assertRaisesRegex(ValueError, "time_s"):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None, :])

    def test_batch_resize_preserves_duration_and_hold_threshold(self):
        rest = np.repeat(self.data.qpos[None, :], 2, axis=0)
        raised = rest.copy()
        for joint in ("ball_tray_joint", "ball_tray_ball_joint"):
            address = int(self.model.jnt_qposadr[self.model.joint(joint).id])
            raised[:, address + 2] += 0.1
        for duration in (5.0, 15.0):
            with self.subTest(duration=duration):
                self.evaluator.reset()
                self.evaluator.configure_from_randomization(
                    self.model,
                    SimpleNamespace(seed=12, metadata={"hold_duration_s": duration}),
                )
                initial = self.evaluator.evaluate_qpos_batch(rest, time_s=0)
                np.testing.assert_array_equal(
                    initial.metrics["hold_duration_s"], [duration, duration]
                )
                self.evaluator.evaluate_qpos_batch(raised, time_s=1)
                before = self.evaluator.evaluate_qpos_batch(
                    raised, time_s=1 + min(11.5, duration - 0.01)
                )
                self.assertFalse(before.metrics["hold_complete"].any())
                reached = self.evaluator.evaluate_qpos_batch(
                    raised, time_s=1 + duration
                )
                self.assertTrue(reached.metrics["hold_complete"].all())
                self.assertFalse(reached.success.any())
                lowered = self.evaluator.evaluate_qpos_batch(
                    rest, time_s=2 + duration
                )
                self.assertTrue(lowered.success.all())

    def test_reset_retains_duration_and_clears_hold_state(self):
        self.sample(1, lifted=True)
        self.sample(6, lifted=True)
        self.assertTrue(self.sample(7).scalar_success())
        self.evaluator.reset(nworld=3)
        qpos = np.repeat(self.data.qpos[None, :], 3, axis=0)
        result = self.evaluator.evaluate_qpos_batch(qpos, time_s=0)
        np.testing.assert_array_equal(result.metrics["hold_duration_s"], [5, 5, 5])
        self.assertFalse(result.metrics["hold_complete"].any())
        self.assertFalse(result.success.any())
        np.testing.assert_array_equal(result.metrics["continuous_hold_s"], [0, 0, 0])

    def test_existing_contact_is_not_a_lowering_event(self):
        address = int(self.model.jnt_qposadr[self.model.joint("ball_tray_joint").id])
        self.data.qpos[address + 3 : address + 7] = [
            np.cos(np.pi / 6),
            np.sin(np.pi / 6),
            0,
            0,
        ]
        self.sample(1, lifted=True)
        reached = self.sample(6, lifted=True)
        self.assertTrue(reached.metrics["tray_table_contact"][0])
        self.assertTrue(reached.metrics["hold_complete"][0])
        self.assertFalse(reached.scalar_success())
        self.assertFalse(self.sample(6, lifted=True).scalar_success())
        self.assertFalse(self.sample(7, lifted=True).scalar_success())
        self.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
        self.assertFalse(self.sample(8, lifted=True).scalar_success())
        self.assertTrue(self.sample(9).scalar_success())

    def test_duration_sampling_is_reproducible_and_in_range(self):
        values = [sample_hold_duration_s(seed) for seed in range(100)]
        self.assertEqual(values, [sample_hold_duration_s(seed) for seed in range(100)])
        self.assertTrue(all(5 <= value <= 15 for value in values))
        self.assertLess(min(values), 6)
        self.assertGreater(max(values), 14)

    def test_half_second_requirements_are_uniform_including_both_endpoints(self):
        values = np.asarray([sample_hold_duration_s(seed) for seed in range(21000)])
        durations, counts = np.unique(values, return_counts=True)
        np.testing.assert_array_equal(durations, np.arange(5.0, 15.5, 0.5))
        # Fixed seeds keep this reproducible; the generous bounds catch biased
        # endpoint buckets without requiring an exact random frequency.
        self.assertTrue(np.all((counts > 800) & (counts < 1200)), counts)

    def test_randomized_duration_is_independent_of_object_scales(self):
        randomizer = BallTrayBalancingRandomizer()
        samples = []
        for seed in range(512):
            mujoco.mj_resetData(self.model, self.data)
            state = randomizer.randomize(self.model, self.data, seed=seed)
            samples.append(
                (
                    state.metadata["tray_scale"],
                    state.metadata["ball_scale"],
                    state.metadata["hold_duration_s"],
                )
            )
        samples = np.asarray(samples)
        np.testing.assert_array_equal(
            np.unique(samples[:, 2]), np.arange(5.0, 15.5, 0.5)
        )
        correlations = np.corrcoef(samples, rowvar=False)[:2, 2]
        self.assertTrue(np.all(np.abs(correlations) < 0.15), correlations)

    def test_randomized_prompt_matches_evaluator_and_runtime_duration(self):
        randomizer = BallTrayBalancingRandomizer()
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        for seed in (0, 1, 4):
            with self.subTest(seed=seed):
                state = randomizer.randomize(self.model, self.data, seed=seed)
                duration = state.metadata["hold_duration_s"]
                prompt = (
                    f"balance the ball in the tray for at least {duration:g} seconds"
                )
                self.assertEqual(state.metadata["prompt"], prompt)
                self.assertEqual(state.metadata["balance_duration_s"], duration)
                self.evaluator.reset()
                self.evaluator.configure_from_randomization(self.model, state)
                result = self.evaluator.evaluate_model_data(self.model, self.data)
                self.assertEqual(result.metrics["hold_duration_s"][0], duration)
                runtime.after_reset(self.model, self.data, randomization=state)
                runtime.on_task_evaluation(result)
                self.assertEqual(runtime.observation_prompt(), prompt)
                self.assertEqual(runtime.debug_state()["display_prompt"], prompt)
                runtime.after_step(self.model, self.data)
                self.assertEqual(runtime.observation_prompt(), prompt)

    def test_runtime_prompts_wait_for_lowering(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(self.model, self.data, randomization=self.randomization)
        runtime.on_task_evaluation(self.sample(0))
        prompt = "balance the ball in the tray for at least 5 seconds"
        self.assertEqual(runtime.observation_prompt(), prompt)
        for timestamp in (1, 6):
            runtime.on_task_evaluation(self.sample(timestamp, lifted=True))
            runtime.after_step(self.model, self.data)
            self.assertEqual(runtime.observation_prompt(), prompt)
        self.assertEqual(runtime.debug_state()["state"], "lowering")
        self.assertFalse(runtime.debug_state()["complete"])
        self.assertEqual(
            runtime.debug_state()["display_prompt"], "Lower the tray onto the table"
        )
        runtime.on_task_evaluation(self.sample(7, ball_inside=False))
        runtime.after_step(self.model, self.data)
        self.assertTrue(runtime.debug_state()["complete"])
        self.assertEqual(runtime.observation_prompt(), prompt)

    def test_env_shares_one_evaluation_per_physics_step_and_after_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "tray.xml"
            mujoco.mj_saveLastXML(str(scene), self.model)
            env = MuJoCoYAMEnv(
                config=RobotSystemConfig(),
                render_cameras=False,
                scene_xml=scene,
                control_decimation=4,
            )
            self.addCleanup(env.close)
            env.set_task("ball_tray_balancing")
            env._task_randomizer = BallTrayBalancingRandomizer()
            evaluate = BallTrayBalancingEvaluator.evaluate_model_data
            # This scene has no robot joints; reset the task objects normally.
            with (
                mock.patch.object(env, "get_init_q", return_value=np.empty(0)),
                mock.patch.object(
                    BallTrayBalancingEvaluator,
                    "evaluate_model_data",
                    autospec=True,
                    side_effect=evaluate,
                ) as score,
            ):
                obs, info = env.reset(seed=4)
                self.assertEqual(score.call_count, 1)
                duration = info["randomization"].metadata["hold_duration_s"]
                self.assertEqual(
                    obs["prompt"],
                    f"balance the ball in the tray for at least {duration:g} seconds",
                )
                self.assertEqual(env.task_runtime_debug_state()["duration_s"], duration)
                _, reward, _, _, step_info = env.step(np.empty(0), render_obs=False)
                self.assertEqual(score.call_count, 5)
                self.assertIs(env._task_runtime._balance_result, env._step_task_eval)
                self.assertEqual(reward, env._step_task_eval.scalar_reward())
                self.assertEqual(
                    step_info["task_success"], env._step_task_eval.scalar_success()
                )
                env.reset(seed=12)
                self.assertEqual(score.call_count, 6)
                self.assertFalse(env._step_task_eval.metrics["hold_complete"][0])
                env.reload_from_model(env.model)
                self.assertIsNone(env._step_task_eval)
                self.assertIsNone(env._task_runtime._balance_result)
                env.reset(seed=4)
                self.assertEqual(score.call_count, 7)
                self.assertIs(env._task_runtime._balance_result, env._step_task_eval)


class ChessTest(unittest.TestCase):
    def setUp(self):
        self.goals = {
            "white_pawn_1_jnt": [0, 0, 0.03],
            "white_pawn_2_jnt": [0.05, 0, 0.03],
            "black_rook_1_jnt": [0.2, 0, 0.03],
        }
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            + body("board", "chessboard", "0 0 0")
            + "".join(
                body(name[:-4], name, " ".join(map(str, xyz)))
                for name, xyz in self.goals.items()
            )
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.evaluator = ChessBoardSetupEvaluator(
            model=self.model, spec=get_task_spec("set up chess pieces on the board")
        )
        self.configure_targets("white_pawn_1_jnt")

    def configure_targets(self, *joints):
        self.evaluator.configure_from_randomization(
            self.model,
            SimpleNamespace(
                object_states={"chessboard": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}},
                metadata={
                    "target_joints": list(joints),
                    "target_poses": {
                        name: {"pos": self.goals[name]} for name in joints
                    },
                    "board_target_poses": {
                        name: {"pos": xyz} for name, xyz in self.goals.items()
                    },
                },
            ),
        )

    def result(self):
        return self.evaluator.evaluate_qpos_batch(self.data.qpos[None, :])

    def test_selected_and_nonselected_same_class_can_swap(self):
        position(
            self.model, self.data, "white_pawn_1_jnt", self.goals["white_pawn_2_jnt"]
        )
        position(
            self.model, self.data, "white_pawn_2_jnt", self.goals["white_pawn_1_jnt"]
        )
        self.assertTrue(self.result().scalar_success())
        self.assertEqual(self.result().metrics["piece_count"], 3)

    def test_nontarget_disturbance_does_not_block_success_or_reduce_reward(self):
        position(self.model, self.data, "white_pawn_2_jnt", [0.5, 0, 0.03])
        result = self.result()
        self.assertTrue(result.scalar_success())
        self.assertEqual(result.reward[0], 1.0)
        self.assertEqual(result.metrics["target_count"], 1)
        self.assertEqual(result.metrics["num_targets_placed"][0], 1)
        self.assertEqual(result.metrics["num_pieces_placed"][0], 2)
        self.assertEqual(result.metrics["num_non_target_disturbed"][0], 1)

    def test_selected_square_must_be_filled(self):
        position(self.model, self.data, "white_pawn_1_jnt", [0.5, 0, 0.03])
        result = self.result()
        self.assertFalse(result.scalar_success())
        self.assertEqual(result.reward[0], 0.0)
        self.assertEqual(result.metrics["num_non_target_disturbed"][0], 0)

    def test_selected_square_takes_priority_over_nonselected_square(self):
        position(self.model, self.data, "white_pawn_1_jnt", [0.5, 0, 0.03])
        # This equivalent pawn is closer to the unselected square, but it can
        # fill the selected square. The unselected square must not consume it.
        position(self.model, self.data, "white_pawn_2_jnt", [0.026, 0, 0.03])
        result = self.result()
        self.assertTrue(result.scalar_success())
        self.assertEqual(result.metrics["targets_placed"], [["white_pawn_1_jnt"]])
        self.assertEqual(result.metrics["num_non_target_disturbed"][0], 1)

    def test_upright_equivalent_piece_can_fill_target_beside_toppled_piece(self):
        address = int(self.model.jnt_qposadr[self.model.joint("white_pawn_1_jnt").id])
        self.data.qpos[address + 3 : address + 7] = [0, 1, 0, 0]
        position(self.model, self.data, "white_pawn_2_jnt", [0.026, 0, 0.03])
        self.assertTrue(self.result().scalar_success())

    def test_reward_counts_only_selected_targets(self):
        self.configure_targets("white_pawn_1_jnt", "white_pawn_2_jnt")
        position(self.model, self.data, "white_pawn_2_jnt", [0.5, 0, 0.03])
        result = self.result()
        self.assertFalse(result.scalar_success())
        self.assertEqual(result.reward[0], 0.5)
        self.assertEqual(result.metrics["num_targets_placed"][0], 1)
        self.assertEqual(result.metrics["num_pieces_placed"][0], 2)

    def test_empty_target_subset_is_unconfigured(self):
        self.configure_targets()
        result = self.result()
        self.assertFalse(result.scalar_success())
        self.assertFalse(result.metrics["goal_configured"])

    def test_selected_subset_is_scored_independently_for_each_world(self):
        qpos = np.repeat(self.data.qpos[None, :], 2, axis=0)
        address = int(self.model.jnt_qposadr[self.model.joint("white_pawn_1_jnt").id])
        qpos[1, address] = 0.5
        result = self.evaluator.evaluate_qpos_batch(qpos)
        np.testing.assert_array_equal(result.success, [True, False])
        np.testing.assert_array_equal(result.reward, [1.0, 0.0])

    def test_matching_finds_feasible_assignment(self):
        self.configure_targets("white_pawn_1_jnt", "white_pawn_2_jnt")
        position(self.model, self.data, "white_pawn_1_jnt", [0.024, 0, 0.03])
        position(self.model, self.data, "white_pawn_2_jnt", [-0.025, 0, 0.03])
        self.assertTrue(self.result().scalar_success())

    def test_different_classes_cannot_swap(self):
        position(
            self.model, self.data, "white_pawn_1_jnt", self.goals["black_rook_1_jnt"]
        )
        position(
            self.model, self.data, "black_rook_1_jnt", self.goals["white_pawn_1_jnt"]
        )
        self.assertFalse(self.result().scalar_success())


class ContactPlacementTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="table_plane" type="plane" size="1 1 .1"/>'
            + body("object", "task_object_joint", "0 -.2 .009")
            + body("left_link_left_finger", "gripper", "2 0 0")
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.evaluator = InhandTransferOtherSideEvaluator(
            model=self.model,
            spec=get_task_spec("inhand transfer the item to other side"),
        )
        self.evaluator.configure_from_randomization(
            self.model,
            SimpleNamespace(
                metadata={"side": "left"},
                object_states={"task_object_joint": {"pos": [0, 0.2, 0.01]}},
            ),
        )

    def result(self):
        mujoco.mj_forward(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def test_requires_table_contact_not_merely_low_height(self):
        self.assertTrue(self.result().scalar_success())
        position(self.model, self.data, "task_object_joint", [0, -0.2, 0.08])
        self.assertFalse(self.result().scalar_success())

    def test_gripper_contact_prevents_success(self):
        position(self.model, self.data, "gripper", [0, -0.2, 0.025])
        result = self.result()
        self.assertTrue(result.metrics["object_table_contact"][0])
        self.assertTrue(result.metrics["object_gripper_contact"][0])
        self.assertFalse(result.scalar_success())


class LegoToleranceTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            + "".join(
                body(
                    f"{color}_bin",
                    f"{color}_lego_sorting_bin_joint",
                    f"{x} 0 .2",
                    f'<site name="{color}_lego_bin_goal_region" type="box" size=".1 .1 .1"/>',
                )
                + body(f"{color}_lego_1", f"{color}_lego_1_joint", f"{x} 0 .2")
                for color, x in (("red", 0), ("yellow", 1), ("blue", 2))
            )
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.evaluator = LegoBlocksSortingEvaluator(
            model=self.model, spec=get_task_spec("lego blocks sorting")
        )

    def result(self, dx, dy=0):
        position(
            self.model,
            self.data,
            "blue_lego_1_joint",
            [2 + 0.1 + dx, 0.1 + dy if dy else 0, 0.2],
        )
        return self.evaluator.evaluate_qpos_batch(self.data.qpos[None, :])

    def test_face_tolerance(self):
        self.assertTrue(self.result(0.0049).scalar_success())
        self.assertFalse(self.result(0.0051).scalar_success())

    def test_euclidean_corner_tolerance(self):
        self.assertFalse(self.result(0.004, 0.004).scalar_success())
        self.assertTrue(self.result(0.0028, 0.004).scalar_success())


class RelativePlacementTest(unittest.TestCase):
    def test_nonnegative_along_with_existing_upper_and_perpendicular_limits(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            + body("mover", "mover_joint", "0 0 .02")
            + body("reference", "reference_joint", "0 0 .02")
            + "</worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        evaluator = PutRelativeEvaluator(
            model=model, spec=get_task_spec("put_relative")
        )
        evaluator.configure_from_randomization(
            model,
            SimpleNamespace(
                metadata={
                    "mover_object": "mover",
                    "reference_object": "reference",
                    "direction_offset_xy": [1, 0],
                },
                object_states={"mover_joint": {"pos": [0, 0, 0.02]}},
            ),
        )
        for x, y, expected in (
            (0, 0, True),
            (0.001, 0, True),
            (-0.001, 0, False),
            (0.301, 0, False),
            (0.1, 0.081, False),
        ):
            with self.subTest(x=x, y=y):
                position(model, data, "mover_joint", [x, y, 0.02])
                self.assertEqual(
                    evaluator.evaluate_qpos_batch(data.qpos[None, :]).scalar_success(),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
