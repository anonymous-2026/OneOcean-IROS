from __future__ import annotations

import argparse
from pathlib import Path


def _replace_in_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    new = text
    for a, b in replacements:
        if a not in new:
            continue
        new = new.replace(a, b)
    if new != text:
        path.write_text(new, encoding="utf-8")


def _rewrite_debug_draw_block(path: Path) -> None:
    """
    Make DebugDraw optional and idempotent.

    MarineGym upstream imports `omni.isaac.debug_draw`, which may be missing in Isaac Sim 5.1.
    This function overwrites the entire DebugDraw section (import + class) with a safe version.
    """

    text = path.read_text(encoding="utf-8")

    # Prefer locating any reference to omni.isaac.debug_draw (handles previously-corrupted states).
    idx_mod = text.find("omni.isaac.debug_draw")
    start = -1
    if idx_mod != -1:
        line_start = text.rfind("\n", 0, idx_mod) + 1
        start = line_start
        # If there are preceding `try:` lines (possibly multiple due to prior edits), include them.
        while True:
            prev_start = text.rfind("\n", 0, start - 1) + 1
            prev_line = text[prev_start:start].strip()
            if prev_line == "try:":
                start = prev_start
                continue
            break
    else:
        start = text.find("from omni.isaac.debug_draw import _debug_draw")
        if start == -1:
            start = text.find("try:\n    from omni.isaac.debug_draw import _debug_draw")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch MarineGym sources for Isaac Sim 5.1 + TorchRL>=0.11.")
    parser.add_argument("--src", type=Path, required=True, help="Path to MarineGym-main (cached copy under runs/_cache).")
    args = parser.parse_args()

    src: Path = args.src.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Missing src: {src}")

    # 1) Make envs import robust (Track/Landing may not exist in this snapshot).
    _replace_in_file(
        src / "marinegym/envs/__init__.py",
        [
            (
                "from .single import Hover, Track, Landing\nfrom .isaac_env import IsaacEnv",
                "from .single import Hover\nfrom .isaac_env import IsaacEnv",
            )
        ],
    )

    # 2) TorchRL spec API compatibility (CompositeSpec/DiscreteTensorSpec moved in torchrl>=0.11).
    _replace_in_file(
        src / "marinegym/envs/isaac_env.py",
        [
            (
                "from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec\n",
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n"
                "from torchrl.data.tensor_specs import TensorSpec, Unbounded\n",
            ),
            (
                '        self.done_spec = CompositeSpec({\n'
                '            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),\n'
                '            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),\n'
                '            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),\n'
                "        }).expand(self.num_envs).to(self.device)\n",
                '        self.done_spec = CompositeSpec({\n'
                '            "done": Unbounded((1,), dtype=torch.bool),\n'
                '            "terminated": Unbounded((1,), dtype=torch.bool),\n'
                '            "truncated": Unbounded((1,), dtype=torch.bool),\n'
                "        }).expand(self.num_envs).to(self.device)\n",
            ),
        ],
    )

    # 2b) Isaac Sim 5.1: omni.isaac.debug_draw may be missing. Make DebugDraw optional (idempotent).
    _rewrite_debug_draw_block(src / "marinegym/envs/isaac_env.py")

    _replace_in_file(
        src / "marinegym/envs/single/hover.py",
        [
            (
                "from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec\n",
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n"
                "from torchrl.data.tensor_specs import UnboundedContinuous as UnboundedContinuousTensorSpec\n",
            )
        ],
    )

    _replace_in_file(
        src / "marinegym/utils/torchrl/env.py",
        [
            (
                "from torchrl.data import TensorSpec, CompositeSpec\n",
                "from torchrl.data.tensor_specs import TensorSpec\n"
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n",
            )
        ],
    )

    _replace_in_file(
        src / "marinegym/robots/drone/underwaterVehicle.py",
        [
            (
                "from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec\n",
                "from torchrl.data.tensor_specs import Bounded as BoundedTensorSpec\n"
                "from torchrl.data.tensor_specs import Composite as CompositeSpec\n"
                "from torchrl.data.tensor_specs import UnboundedContinuous as UnboundedContinuousTensorSpec\n",
            ),
            (
                "        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)\n",
                "        self.action_spec = BoundedTensorSpec(-1, 1, shape=(self.num_rotors,), device=self.device)\n",
            ),
        ],
    )

    # 3) Python 3.11 dataclass strictness: mutable defaults must use default_factory.
    _replace_in_file(
        src / "marinegym/robots/config.py",
        [
            ("from dataclasses import dataclass\n", "from dataclasses import dataclass, field\n"),
            (
                "class RobotCfg:\n"
                "    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg()\n"
                "    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg()\n",
                "class RobotCfg:\n"
                "    rigid_props: RigidBodyPropertiesCfg = field(default_factory=RigidBodyPropertiesCfg)\n"
                "    articulation_props: ArticulationRootPropertiesCfg = field(default_factory=ArticulationRootPropertiesCfg)\n",
            ),
        ],
    )

    # 4) Isaac Sim 5.1: SimulationContext private attributes changed; avoid hard dependency on _physics_sim_view.
    _replace_in_file(
        src / "marinegym/robots/robot.py",
        [
            (
                "        if SimulationContext.instance()._physics_sim_view is not None:\n",
                "        if getattr(SimulationContext.instance(), \"_physics_sim_view\", None) is not None:\n",
            )
            ,
            (
                "        if SimulationContext.instance()._physics_sim_view is None:\n",
                "        if getattr(SimulationContext.instance(), \"_physics_sim_view\", \"missing\") is None:\n",
            ),
        ],
    )

    # 5) Isaac Sim 5.1: view wrappers need to tolerate missing SimulationContext._physics_sim_view and new get_world_poses API.
    _replace_in_file(
        src / "marinegym/views/__init__.py",
        [
            (
                "        if SimulationContext.instance()._physics_sim_view is None:\n",
                "        if getattr(SimulationContext.instance(), \"_physics_sim_view\", \"missing\") is None:\n",
            ),
            (
                "    def get_world_poses(\n"
                "        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True\n"
                "    ) -> Tuple[torch.Tensor, torch.Tensor]:\n",
                "    def get_world_poses(\n"
                "        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True, usd: bool = False\n"
                "    ) -> Tuple[torch.Tensor, torch.Tensor]:\n",
            ),
        ],
    )

    print(f"[patch] patched MarineGym at: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
