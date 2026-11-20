# UR magnet IK and Jacobian validation guide

This guide accompanies `ur_magnet_ik_jacobian_validation.py`. The script is a
stationary, read-only development tool: it reads robot state and calls
kinematics functions, but it cannot command motion.

## 1. What the script proves—and what it does not

The script checks four separate questions:

1. Is the independent standard-DH forward kinematics implemented correctly?
2. Does its analytic geometric Jacobian match central finite differences?
3. Does the independent damped inverse-Jacobian IK agree with `ur_rtde` IK?
4. How closely does the nominal or corrected DH model agree with the individual
   robot controller's calibrated kinematics?

It does **not** independently measure the physical magnet unless you supply a
camera/tracker result in the robot-base frame. By default, the magnet pose is
inferred from the robot's measured TCP:

\[
{}^{R}T_M = {}^{R}T_{TCP}\,{}^{TCP}T_M.
\]

This result contains encoder-based robot pose information and your calibrated
tool-to-magnet transform. It cannot reveal an error in `T_TCP_M` by itself. To
validate that physical transform, use an external measurement:

\[
{}^{R}T_{M,\text{measured}}
  = {}^{R}T_C\,{}^{C}T_{M,\text{measured}}.
\]

Here `C` is the calibrated camera/tracker frame.

## 2. The transform chain

The notation `T_A_B` means “pose of B expressed in A” and maps a B-coordinate
point into A:

\[
p_A = {}^{A}T_B p_B.
\]

The complete chain is:

| Transform | Meaning | Source |
|---|---|---|
| `T_R_F(q)` | output flange in UR base | independent DH or UR calibrated FK |
| `T_F_TCP` | active TCP in flange | read from `getTCPOffset()` |
| `T_TCP_M` | magnet centre/body frame in active TCP | physical calibration |
| `T_F_M` | magnet in flange | `T_F_TCP @ T_TCP_M` |
| `T_R_M` | magnet in robot base | `T_R_F(q) @ T_F_M` |

There is no additional `z_offset`. Any physical translation and orientation
between TCP and magnet belongs in `T_TCP_M` exactly once.

## 3. Configuration order

Edit only the configuration section at the top of the script.

### Step 1: select the exact robot

Set `robot_model` to the real arm, for example `"ur5e"`. Do not select the
closest-looking model. The link parameters differ between e-Series and CB
robots.

The built-in table is the public nominal standard-DH model. A real controller
uses factory calibration. Therefore:

- analytic DH versus its own finite difference should agree very closely;
- nominal DH versus UR calibrated kinematics can have a real residual;
- a residual in the second comparison is not automatically a programming bug.

If controller-specific corrections are available, enter them in the four
`dh_delta_*` vectors. Preserve joint order:

`base, shoulder, elbow, wrist1, wrist2, wrist3`.

### Step 2: calibrate `T_TCP_M`

`T_tcp_magnet_pose6` is:

`[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]`

and describes the magnet frame in the **active TCP**. The first three values
place the magnet centre; the rotation vector aligns the magnet's body axes.
Include holder geometry and any deliberate magnetic-axis convention.

Only set `assume_tcp_is_magnet_frame=True` if the active TCP has physically
been established at the magnet centre and its axes really match the defined
magnet frame.

### Step 3: replace the joint bounds

The configured ±2π values are generic algorithm bounds. Replace them with the
limits allowed by the installed robot and cell. These bounds guide numerical
IK; they do not replace UR safety planes, joint safety limits, collision
checking, or a later QP controller.

### Step 4: start offline

Keep `use_live_robot=False`. Fill `offline_actual_q_rad`, use a known TCP
offset, and run the script. In this mode the UR comparison is absent, but the
independent transform, Jacobian, IK, logging, and plotting paths run.

The supplied test file exercises these mathematics with a known synthetic
model.

### Step 5: make a stationary live comparison

Set `use_live_robot=True` only after the preceding checks pass. The script asks
for an exact confirmation before creating `RTDEControlInterface`.

Although the script has no motion method, creation of that interface may upload
an RTDE control script. The robot should be stationary, and another production
program should not be running.

## 4. Forward kinematics and analytic Jacobian

For each UR revolute joint, the script uses the standard DH transform

\[
{}^{i-1}T_i = R_z(\theta_i)T_z(d_i)T_x(a_i)R_x(\alpha_i).
\]

Before applying each joint transform it records the joint origin \(o_i\) and
axis \(z_i\), both expressed in robot base R. For the magnet-centre position
\(p_M\), the geometric Jacobian column is

\[
J_i =
\begin{bmatrix}
z_i \times (p_M-o_i)\\
z_i
\end{bmatrix}.
\]

Thus

\[
\begin{bmatrix}v_M\\\omega_M\end{bmatrix}
= J_M(q)\dot q.
\]

The row convention is `[vx, vy, vz, wx, wy, wz]`, expressed in robot base R.
Linear rows have units m/rad and angular rows rad/rad.

The central-difference check perturbs every physical joint by the configured
angle \(h\):

\[
\frac{\partial p}{\partial q_i}
\approx \frac{p(q+h e_i)-p(q-h e_i)}{2h}.
\]

Orientation differences use the rotation logarithm of
\(R(q+h e_i)R(q-h e_i)^T\), so the angular derivative is also expressed in the
robot-base frame.

## 5. Independent inverse kinematics

The pose error is

\[
e = \begin{bmatrix}p_d-p(q)\\
\log(R_dR(q)^T)\end{bmatrix}.
\]

Because metres and radians cannot be compared directly, angular rows are
scaled by the configured characteristic length \(L_c\):

\[
S=\operatorname{diag}(1,1,1,L_c,L_c,L_c).
\]

Each iteration uses damped least squares:

\[
\Delta q = (SJ)^T\left[(SJ)(SJ)^T+\lambda^2I\right]^{-1}Se.
\]

The script then applies:

1. per-joint step limits;
2. a total joint-step norm limit;
3. configured joint-position bounds;
4. a backtracking line search;
5. increased damping if the trial step does not reduce pose error.

The seed is the measured joint state. `ur_rtde` IK is also given that state as
`qnear`, encouraging both solvers to choose the same branch.

The UR inverse-kinematics function works on the active TCP, not the magnet
frame. Therefore the requested magnet target is converted exactly as

\[
{}^{R}T_{TCP,d} = {}^{R}T_{M,d}({}^{TCP}T_M)^{-1}.
\]

The script does not call `setTcp()`.

## 6. How to interpret disagreement

Use this order:

1. **Own analytic J versus own finite-difference J.** A large error indicates
   a transform, row-order, frame, or derivative implementation problem.
2. **UR Jacobian versus finite differences of UR FK.** A large error indicates
   an API/version, TCP-offset, row-order, or numerical-step issue.
3. **Own J versus UR J.** Once both self-checks pass, the residual mainly shows
   nominal-versus-calibrated robot model differences.
4. **IK terminal pose errors.** Judge both joint vectors by forward pose, not
   joint equality alone. A six-axis arm can have multiple valid IK branches.
5. **Wrapped joint difference.** This removes harmless ±2π representation
   changes but still exposes a different IK branch.

The raw 6×6 Frobenius norm mixes metres and radians, so the report also stores
separate linear/angular errors and scaled condition numbers.

## 7. Output files

Each run creates a timestamped directory containing:

- `summary.json`: complete transforms, configurations, IK histories,
  Jacobians, comparisons, and pass/fail checks;
- `ik_comparison.csv`: one row per target and the two solutions' residuals;
- `jacobian_summary.csv`: condition numbers and matrix-level errors;
- `jacobian_elements.csv`: every element of every Jacobian;
- `validation_overview.png`: arm/target geometry, IK joint displacement,
  Jacobian difference, and IK terminal residuals.

## 8. Recommended development sequence toward a controller

Do not command the new IK output yet. Progress through these stages:

1. stationary offline analytic-versus-finite-difference validation;
2. stationary live UR FK/Jacobian/IK comparison;
3. external measurement of `T_R_M` to validate `T_TCP_M`;
4. small, manually reviewed joint targets checked by UR safety functions;
5. slow supervised motion with independent stop capability;
6. only then place the robot Jacobian inside the magnetic-beam controller.

For the later joint-space QP, combine the magnetic model sensitivity and robot
Jacobian. If beam output \(x_b\) depends on magnet pose \(x_M\), then locally

\[
\dot x_b = J_{bM}J_{Mq}\dot q.
\]

The QP decision variable should be joint velocity or joint increment, allowing
joint position, joint velocity, workspace, trust-region, and collision
constraints to be imposed before any command is sent. The validator created
here supplies and tests the `J_Mq` part of that chain.

## 9. Physical checks still required

- The installed robot model and software version match the configuration.
- The active TCP read from the controller is the TCP used to define `T_TCP_M`.
- `T_TCP_M` includes the true magnet centre and correct magnetic/body axes.
- The UR base frame matches the frame used by the beam simulation.
- Any camera-derived `T_R_M` has a calibrated `T_R_C` and time synchronization.
- Joint bounds match the cell, and separate collision/safety constraints exist.
- The robot remains stationary during the snapshot.
- The installed `ur_rtde`/PolyScope version supports the queried Jacobian API.

The supplied code was compiled and tested with offline synthetic kinematics. It
was not executed against the physical robot or camera.
