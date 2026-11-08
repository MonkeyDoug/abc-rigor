"""CPU regressions for the revised simulator success conditions."""

import unittest
from types import SimpleNamespace

import mujoco

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

    def test_duration_sampling_is_reproducible_and_in_range(self):
        values = [sample_hold_duration_s(seed) for seed in range(100)]
        self.assertEqual(values, [sample_hold_duration_s(seed) for seed in range(100)])
        self.assertTrue(all(5 <= value <= 15 for value in values))
        self.assertLess(min(values), 6)
        self.assertGreater(max(values), 14)

    def test_runtime_prompts_wait_for_lowering(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(self.model, self.data, randomization=self.randomization)
        for timestamp in (1, 6):
            self.sample(timestamp, lifted=True)
            runtime.after_step(self.model, self.data)
        self.assertEqual(runtime.debug_state()["state"], "lowering")
        self.assertFalse(runtime.debug_state()["complete"])
        self.sample(7, ball_inside=False)
        runtime.after_step(self.model, self.data)
        self.assertTrue(runtime.debug_state()["complete"])


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
        self.evaluator.configure_from_randomization(
            self.model,
            SimpleNamespace(
                object_states={"chessboard": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}},
                metadata={
                    "target_joints": ["white_pawn_1_jnt"],
                    "target_poses": {
                        "white_pawn_1_jnt": {"pos": self.goals["white_pawn_1_jnt"]}
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

    def test_nontarget_piece_must_also_be_placed(self):
        position(self.model, self.data, "white_pawn_2_jnt", [0.5, 0, 0.03])
        self.assertFalse(self.result().scalar_success())

    def test_matching_finds_feasible_assignment(self):
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
