# Dynamics and Pollution Model Scope

This document maps the benchmark's engineering models to the public configuration fields. These models provide deterministic, configurable task dynamics; they are not a vehicle-identified hydrodynamic simulator or a field-validated pollution forecast.

## Core frame and current coupling

The core state uses world position `p = [x, y, z] = [east, depth-down, north]`. Dataset current is `c = [u, 0, v]`. The 3DoF/6DoF modes keep a body-relative velocity

`nu = [u_b, v_b, w_b, p_b, q_b, r_b]`.

For diagonal mass `M`, damping `D`, and tracking gains, the implemented update is

`tau_linear = Kp_linear (nu_cmd_linear - nu_linear)`

`tau_angular = Kp_angular (nu_cmd_angular - nu_angular) - Kd_angular nu_angular`

`M nu_dot = tau - D nu`

`p_dot = R(roll, yaw, pitch) nu_linear + c(p, t)`.

`nu_cmd_linear` is the commanded world-relative velocity transformed into the body frame. The angular command aligns yaw with horizontal motion and stabilizes roll/pitch. Integration uses the configured `dt_s`; linear speed, angular rate, depth, tilt, tile bounds, terrain validity, and seafloor clearance are clipped or rejected by the shared constraint path.

The model intentionally omits identified added mass, Coriolis terms, restoring forces, thruster allocation, actuator lag, waves on a vehicle hull, and two-way fluid coupling. The current enters as an externally sampled transport velocity rather than a solved hydrodynamic load.

## Dynamics modes

| Mode | State update | Intended use |
|---|---|---|
| `kinematic` | Directly integrates commanded relative velocity plus current | Fast debugging and contract checks |
| `3dof` | Uses the diagonal velocity model but suppresses roll and pitch; translation and yaw remain | Stable planar-attitude ablation |
| `6dof` | Integrates three relative linear and three angular rates with bounded attitude | Default paper-facing engineering dynamics |

The name `6dof` refers to the six-component body velocity and pose update, not to a claim of high-fidelity vehicle hydrodynamics.

## Default dynamics parameters

| `EnvConfig` field | Default | Meaning |
|---|---:|---|
| `dt_s` | `1.0` | outer simulation step, seconds |
| `max_speed_mps` | `1.2` | relative linear speed cap |
| `dyn_mass_linear` | `(12, 12, 12)` | diagonal linear inertia proxy |
| `dyn_mass_angular` | `(6, 6, 6)` | diagonal angular inertia proxy |
| `dyn_damping_linear` | `(8, 8, 8)` | diagonal linear damping |
| `dyn_damping_angular` | `(3, 3, 3)` | diagonal angular damping |
| `dyn_kp_linear` | `(18, 18, 18)` | linear velocity tracking gain |
| `dyn_kp_angular` | `(10, 10, 10)` | angular-rate tracking gain |
| `dyn_kd_angular` | `(2, 2, 2)` | angular-rate damping gain |
| `dyn_max_angular_rate_rps` | `(1.2, 1.2, 1.2)` | angular-rate cap, rad/s |
| `dyn_yaw_rate_kp` | `2.0` | yaw-error to desired yaw-rate gain |
| `dyn_attitude_rate_kp` | `2.0` | roll/pitch stabilization gain |
| `dyn_max_tilt_rad` | `0.6` | roll/pitch safety limit |

All fields are serialized in `run_meta.json` and `spec_snapshot.json`. They can be overridden by constructing `EnvConfig`; paper CLI flags expose the mode and the environment-level current/constraint settings.

## Pollution observations

Each agent receives a scalar probe

`o_i^c(t) = C(p_i(t), t)`.

The field is synthetic and regeneratable under the recorded seed. It supplies controlled task signals and does not represent in-situ pollution labels.

### Analytic Gaussian

The lightweight field is

`C(p,t) = m(t) C_peak exp(-||p - mu(t)||^2 / (2 sigma^2))`,

with `sigma_m=35`, `C_peak=1`, and a center `mu(t)` advected by the sampled horizontal current. For the internal continuous-containment task, an agent within the default 35 m sink radius applies

`m(t + dt) = m(t) max(0, 1 - 0.12 dt)`.

The reported `mass_fraction` is `m(t)/m(0)`.

### OCPNet 3D field

The OCPNet-backed option numerically advances an advection--diffusion--reaction-style concentration field with a point source and a local agent sink. Its benchmark wrapper uses:

| `OCPNetConfig` field | Default |
|---|---:|
| `grid_resolution` | `(28, 28, 10)` |
| `time_step_s` | `2.0` |
| `diffusion_coefficient` | `8e-8` |
| `decay_rate` | `0.0` |
| `emission_rate` | `0.02` |
| `sink_radius_m` | `8.0` |
| `sink_strength_per_s` | `0.15` |

The 2D dataset current is resampled and repeated over model depth, with zero vertical current. This is an explicit engineering approximation rather than a measured 3D velocity profile. The sink multiplies cells in the configured agent neighborhood by `max(0, 1 - sink_strength_per_s × time_step_s)`.

## Task semantics are not interchangeable

- `pollution_localization` evaluates distance to a hidden concentration source.
- `pollution_containment_multiagent` is an internal continuous-field task that uses sink-driven remaining mass.
- `surface_pollution_cleanup_multiagent` is a canonical discrete-source service task: assigned agents must remain within the task radius for `cleanup_dwell_s`, after which that source is marked complete. Medium difficulty uses 12 sources, 8 s dwell, and one required agent; hard difficulty uses 6 sources, 6 s dwell, and two required agents.

The canonical surface-cleanup results therefore measure source-assignment and service completion, not removed concentration mass. Changing them to continuous-field cleanup would change the benchmark semantics and require a new experiment version; the current release documents the distinction instead of retroactively relabeling existing results.
