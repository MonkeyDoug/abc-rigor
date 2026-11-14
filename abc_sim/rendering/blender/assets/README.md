# YAM D405 rendering assets

`d405_crank_bracket.stl` and `d405_cable_holder.stl` are from
[i2rt v1.3.1](https://github.com/i2rt-robotics/i2rt/tree/1276f63d640eb45c226efd3dc08430b810372e94),
under the MIT license in `LICENSE.i2rt`. Their source files are
`i2rt/robot_models/arm/yam/v1/assets/d405/camera_bracket_for_D405_crank_4310_.stl`
and `Cable_holder.stl`.

The meshes are rigidly transformed into the camera optical frame using the
camera and part poses in `yam_crank_4310_d405.xml` from that revision. Units are
metres; axes are +X right, +Y down, +Z forward. The transform applied to each
original mesh is `p_optical = R(quat_wxyz) @ p_original + position`:

| Asset | Position | Quaternion (w, x, y, z) |
| --- | --- | --- |
| Bracket | `(0.0336000156, 0.03074253708184603, 0.012489905621371325)` | `(0.24185357034749658, 0.6644598185795705, 0.6644607069573393, -0.2418511296502983)` |
| Cable holder | `(0.0260004774, -0.05593873535809368, -0.07889293232763495)` | `(0.5963640034269857, -0.3799331511233836, -0.3799357263946926, -0.5963636641370196)` |

These are rendering accessories for the ABC YAM crank-gripper assembly. They do
not define collision geometry, inertia, camera calibration, or control frames.
The dark bracket finish is an authored material. D405 housing and front-panel
materials follow the [manufacturer's product reference](https://www.realsenseai.com/depth-camera-d405/);
the rounded panel and lens details are visual approximations. The D405 case is
42 × 42 × 23 mm and the stereo baseline is 18 mm.

ABC's older gripper and recorded camera pose leave a roughly 4 mm gap at the
CAD bracket foot. The renderer adds a short, authored adapter spacer between
that foot and the gripper seat. It preserves the recorded camera pose and
does not claim that this adapter is an i2rt production part.
