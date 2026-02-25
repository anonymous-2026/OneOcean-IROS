from __future__ import annotations

import argparse
from pathlib import Path


def _replace_in_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    new = text
    for a, b in replacements:
        if a in new:
            new = new.replace(a, b)
    if new != text:
        path.write_text(new, encoding="utf-8")


def _rewrite_block_between(path: Path, *, start_pat: str, end_pat: str, new_block: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start = text.find(start_pat)
    if start == -1:
        return
    end = text.find(end_pat, start)
    if end == -1:
        return
    new = text[:start] + new_block + text[end:]
    if new != text:
        path.write_text(new, encoding="utf-8")


def _rewrite_debug_draw_block(path: Path) -> None:
    """
    Isaac Sim 5.1 may not ship omni.isaac.debug_draw.
    Overwrite the debug draw import+class with an optional safe version (idempotent).
    """

    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")

    idx_mod = text.find("omni.isaac.debug_draw")
    start = -1
    if idx_mod != -1:
        start = text.rfind("\n", 0, idx_mod) + 1
        while True:
            prev_start = text.rfind("\n", 0, start - 1) + 1
            if text[prev_start:start].strip() == "try:":
                start = prev_start
                continue
            break
    else:
        start = text.find("from omni.isaac.debug_draw import _debug_draw")
        if start == -1:
            start = text.find("class DebugDraw:")
    if start == -1:
        return

    end = text.find("\n\nclass IsaacEnv", start)
    if end == -1:
        end = text.find("\nclass IsaacEnv", start)
    if end == -1:
        return

    block = (
        "try:\n"
        "    from omni.isaac.debug_draw import _debug_draw  # type: ignore\n"
        "except Exception:\n"
        "    _debug_draw = None\n"
        "\n"
        "\n"
        "class DebugDraw:\n"
        "    def __init__(self):\n"
        "        self._draw = None\n"
        "        if _debug_draw is not None:\n"
        "            try:\n"
        "                self._draw = _debug_draw.acquire_debug_draw_interface()\n"
        "            except Exception:\n"
        "                self._draw = None\n"
        "\n"
        "    def clear(self):\n"
        "        if self._draw is None:\n"
        "            return\n"
        "        self._draw.clear_lines()\n"
        "\n"
        "    def plot(self, x: torch.Tensor, size=2.0, color=(1.0, 1.0, 1.0, 1.0)):\n"
        "        if self._draw is None:\n"
        "            return\n"
        "        if not (x.ndim == 2 and x.shape[1] == 3):\n"
        "            raise ValueError(\"x must be a tensor of shape (N, 3).\")\n"
        "        x = x.cpu()\n"
        "        point_list_0 = x[:-1].tolist()\n"
        "        point_list_1 = x[1:].tolist()\n"
        "        sizes = [size] * len(point_list_0)\n"
        "        colors = [color] * len(point_list_0)\n"
        "        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)\n"
        "\n"
        "    def vector(self, x: torch.Tensor, v: torch.Tensor, size=2.0, color=(0.0, 1.0, 1.0, 1.0)):\n"
        "        if self._draw is None:\n"
        "            return\n"
        "        x = x.cpu().reshape(-1, 3)\n"
        "        v = v.cpu().reshape(-1, 3)\n"
        "        if x.shape != v.shape:\n"
        "            raise ValueError(f\"x and v must have the same shape, got {x.shape} and {v.shape}.\")\n"
        "        point_list_0 = x.tolist()\n"
        "        point_list_1 = (x + v).tolist()\n"
        "        sizes = [size] * len(point_list_0)\n"
        "        colors = [color] * len(point_list_0)\n"
        "        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)\n"
        "\n"
        "\n"
    )

    new = text[:start] + block + text[end:]
    if new != text:
        path.write_text(new, encoding="utf-8")


def _patch_underwatervehicle_make_functional(path: Path) -> None:
    """
    tensordict>=0.11 removed make_functional; also functorch vmap pathway is brittle.
    Replace rotor functionalization + vmap step with an explicit per-env rotor TensorDict state and vectorized update.
    """

    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "tensordict>=0.11 removed make_functional" in text:
        return

    start_pat = "        rotor_params = make_functional(self.rotors)\n"
    end_pat = "        self.thrusts = torch.zeros"
    if start_pat in text and end_pat in text:
        new_block = (
            "        # tensordict>=0.11 removed make_functional; keep an explicit per-env rotor state TensorDict instead.\n"
            "        self.TIME_CONSTANTS_0 = torch.tensor(\n"
            "            self.params[\"rotor_configuration\"][\"time_constants\"], device=self.device\n"
            "        )\n"
            "        self.FORCE_CONSTANTS_0 = torch.tensor(\n"
            "            self.params[\"rotor_configuration\"][\"force_constants\"], device=self.device\n"
            "        )\n"
            "        self.rotor_params = TensorDict(\n"
            "            {\n"
            "                \"force_constants\": self.FORCE_CONSTANTS_0.expand(*self.shape, self.num_rotors).clone(),\n"
            "                \"time_constants\": self.TIME_CONSTANTS_0.expand(*self.shape, self.num_rotors).clone(),\n"
            "                \"tau_up\": self.rotors.tau_up.detach().clone().expand(*self.shape, self.num_rotors),\n"
            "                \"tau_down\": self.rotors.tau_down.detach().clone().expand(*self.shape, self.num_rotors),\n"
            "                \"throttle\": torch.zeros(*self.shape, self.num_rotors, device=self.device),\n"
            "                \"directions\": self.rotors.directions.detach().clone().expand(*self.shape, self.num_rotors),\n"
            "                \"rpm\": torch.zeros(*self.shape, self.num_rotors, device=self.device),\n"
            "            },\n"
            "            batch_size=self.shape,\n"
            "            device=self.device,\n"
            "        )\n"
            "\n"
            "        self.tau_up = self.rotor_params[\"tau_up\"]\n"
            "        self.tau_down = self.rotor_params[\"tau_down\"]\n"
            "        self.throttle = self.rotor_params[\"throttle\"]\n"
            "        self.directions = self.rotor_params[\"directions\"]\n"
            "\n"
        )
        _rewrite_block_between(path, start_pat=start_pat, end_pat=end_pat, new_block=new_block)
        text = path.read_text(encoding="utf-8")

    vmap_pat = (
        "        thrusts, moments = vmap(vmap(self.rotors, randomness=\"different\"), randomness=\"same\")(\n"
        "            rotor_cmds, self.rotor_params\n"
        "        )\n"
    )
    if vmap_pat in text:
        vec = (
            "        # Vectorized T200 step using per-env rotor_params.\n"
            "        rotor_cmds = torch.clamp(rotor_cmds, -1, 1)\n"
            "        throttle = self.rotor_params[\"throttle\"]\n"
            "        tau = torch.where(\n"
            "            rotor_cmds > throttle, self.rotor_params[\"tau_up\"], self.rotor_params[\"tau_down\"]\n"
            "        )\n"
            "        tau = torch.clamp(tau, 0, 1)\n"
            "        throttle.add_(tau * (rotor_cmds - throttle))\n"
            "\n"
            "        target_rpm = torch.where(\n"
            "            throttle > 0.075,\n"
            "            3.6599e03 * throttle + 3.4521e02,\n"
            "            torch.where(\n"
            "                throttle < -0.075,\n"
            "                3.4944e03 * throttle - 4.3350e02,\n"
            "                torch.zeros_like(throttle),\n"
            "            ),\n"
            "        )\n"
            "        alpha = torch.exp(-self.dt / self.rotor_params[\"time_constants\"])\n"
            "        rpm = self.rotor_params[\"rpm\"]\n"
            "        rpm.copy_(torch.clamp(alpha * rpm + (1 - alpha) * target_rpm, -3900.0, 3900.0))\n"
            "\n"
            "        fc = self.rotor_params[\"force_constants\"]\n"
            "        thrusts = fc / 4.4e-7 * 9.81 * torch.where(\n"
            "            rpm > 0,\n"
            "            4.7368e-07 * torch.square(rpm) - 1.9275e-04 * rpm + 8.4452e-02,\n"
            "            -3.8442e-07 * torch.square(rpm) - 1.6186e-04 * rpm - 3.9139e-02,\n"
            "        )\n"
            "        moments = thrusts * (-self.directions) * 0.0\n"
        )
        text = text.replace(vmap_pat, vec)

    path.write_text(text, encoding="utf-8")


def _patch_underwatervehicle_multiagent_hydro(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "H6_PATCH_MULTIAGENT_HYDRO" in text:
        return

    start_pat = "    def apply_hydrodynamic_forces(self, flow_vels_w) -> TensorDict:\n"
    end_pat = "\n    def calculate_acc"
    new_block = (
        "    def apply_hydrodynamic_forces(self, flow_vels_w) -> TensorDict:\n"
        "\n"
        "        # H6_PATCH_MULTIAGENT_HYDRO: support shape=(num_envs, num_robots) (multi-agent)\n"
        "        body_vels = self.vel_b.clone()\n"
        "        body_rpy = quaternion_to_euler(self.rot)\n"
        "        flow_vels_b = torch.cat([\n"
        "            quat_rotate_inverse(self.rot, flow_vels_w[..., :3]),\n"
        "            quat_rotate_inverse(self.rot, flow_vels_w[..., 3:])\n"
        "        ], dim=-1)\n"
        "        body_vels -= flow_vels_b\n"
        "        body_vels[..., [1,2,4,5]] *= -1\n"
        "        body_rpy[..., [1,2]] *= -1\n"
        "\n"
        "        body_acc = self.calculate_acc(body_vels)\n"
        "        damping = self.calculate_damping(body_vels)\n"
        "        added_mass = self.calculate_added_mass(body_acc)\n"
        "        coriolis = self.calculate_corilis(body_vels)\n"
        "        buoyancy = self.calculate_buoyancy(body_rpy)\n"
        "\n"
        "        hydro = - (added_mass + coriolis + damping)\n"
        "        hydro[..., [1,2,4,5]] *= -1\n"
        "        buoyancy[..., [1,2,4,5]] *= -1\n"
        "\n"
        "        return hydro[..., 0:3] + buoyancy[..., 0:3], hydro[..., 3:6] + buoyancy[..., 3:6]\n"
        "\n"
    )
    _rewrite_block_between(path, start_pat=start_pat, end_pat=end_pat, new_block=new_block)


def patch_marinegym(src: Path) -> None:
    src = src.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    _replace_in_file(
        src / "marinegym/envs/__init__.py",
        [
            (
                "from .single import Hover, Track, Landing\nfrom .isaac_env import IsaacEnv",
                "from .single import Hover\nfrom .isaac_env import IsaacEnv",
            )
        ],
    )

    _replace_in_file(
        src / "marinegym/envs/isaac_env.py",
        [
            (
                "from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec\n",
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n"
                "from torchrl.data.tensor_specs import TensorSpec, Unbounded\n",
            ),
        ],
    )
    _rewrite_debug_draw_block(src / "marinegym/envs/isaac_env.py")

    _replace_in_file(
        src / "marinegym/robots/drone/underwaterVehicle.py",
        [
            (
                "from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec\n",
                "from torchrl.data.tensor_specs import Bounded as BoundedTensorSpec\n"
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n"
                "from torchrl.data.tensor_specs import UnboundedContinuous as UnboundedContinuousTensorSpec\n",
            ),
        ],
    )
    _patch_underwatervehicle_make_functional(src / "marinegym/robots/drone/underwaterVehicle.py")
    _patch_underwatervehicle_multiagent_hydro(src / "marinegym/robots/drone/underwaterVehicle.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch MarineGym cached sources for Isaac Sim 5.1 / Py3.11.")
    parser.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()
    patch_marinegym(args.src)
    print(f"[patch] patched MarineGym at: {Path(args.src).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

