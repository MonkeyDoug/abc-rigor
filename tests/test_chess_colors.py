"""Chess identities, requested colors and rendered materials must agree."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from abc_sim.randomization.tasks.chess import _CHESS_PIECE_JOINTS, ChessRandomizer


class ChessColorsTest(unittest.TestCase):
    def test_both_scenes_match_identities_and_selection_to_visual_colors(self):
        models = Path(__file__).resolve().parents[1] / "abc_sim" / "models"
        randomizer = ChessRandomizer()
        for scene in ("yam_chess_scene.xml", "yam_flexible_chess_scene.xml"):
            with self.subTest(scene=scene):
                root = ET.parse(models / scene).getroot()
                materials = {}
                positions = {"white": [], "black": []}
                for body in root.findall(".//body"):
                    joint = body.find("freejoint")
                    if joint is None or joint.get("name") not in _CHESS_PIECE_JOINTS:
                        continue
                    name = joint.get("name")
                    color = randomizer._piece_color(name)
                    self.assertEqual(body.get("name"), name.removesuffix("_jnt"))
                    geom = body.find("geom")
                    materials[name] = geom.get("material")
                    self.assertEqual(
                        materials[name],
                        "light_piece" if color == "white" else "dark_piece",
                    )
                    positions[color].append(float(body.get("pos").split()[0]))
                self.assertEqual(set(materials), set(_CHESS_PIECE_JOINTS))
                # The robot-side pieces remain visually white.
                self.assertLess(max(positions["white"]), min(positions["black"]))
                for color, material in (
                    ("white", "light_piece"),
                    ("black", "dark_piece"),
                ):
                    for seed in range(10):
                        selected = randomizer._sample_target_joints(
                            np.random.default_rng(seed), f"{color}_only", 10
                        )
                        self.assertEqual(len(selected), 10)
                        self.assertTrue(
                            all(materials[name] == material for name in selected)
                        )


if __name__ == "__main__":
    unittest.main()
