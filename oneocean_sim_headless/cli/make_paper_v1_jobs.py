#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate JSONL jobs for H1 paper_v1 runs (heuristic/BC/LLM pool).")
    ap.add_argument("--drift-npz", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True, help="Jobs will write into out_root/<method_id>/")
    ap.add_argument("--jobs-jsonl", type=str, default="", help="Output JSONL path (default: out_root/jobs_<stamp>.jsonl)")
    ap.add_argument("--preset", type=str, default="paper_v1", choices=["paper_v1", "paper_v1_llm"], help="run_matrix preset to use.")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--dynamics-model", type=str, default="6dof", choices=["kinematic", "3dof", "6dof"])
    ap.add_argument("--constraint-mode", type=str, default="hard", choices=["off", "hard"])
    ap.add_argument("--bathy-mode", type=str, default="hard", choices=["off", "hard"])
    ap.add_argument("--seafloor-clearance-m", type=float, default=1.0)
    ap.add_argument("--current-gain", type=float, default=1.0)
    ap.add_argument("--rec-step-stride", type=int, default=5, help="Pass through to run_matrix (reduce I/O); 1 keeps full-rate recordings.")
    ap.add_argument("--bc-weights-npz", type=str, default="", help="If provided, include an mlp_bc job.")
    ap.add_argument("--llm-cache-root", type=str, default="", help="Cache root for LLM JSON outputs (optional).")
    ap.add_argument("--llm-call-stride-steps", type=int, default=30)
    ap.add_argument("--llm-max-new-tokens", type=int, default=192)
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM jobs (heuristic/BC only).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_path = Path(args.jobs_jsonl).expanduser().resolve() if str(args.jobs_jsonl).strip() else (out_root / f"jobs_{stamp}.jsonl")

    drift_npz = str(Path(args.drift_npz).expanduser().resolve())
    llm_cache_root = str(Path(args.llm_cache_root).expanduser().resolve()) if str(args.llm_cache_root).strip() else ""

    base = [
        # Always use the current interpreter so jobs run in the same environment (deps installed).
        sys.executable,
        "-m",
        "oneocean_sim_headless.cli.run_matrix",
        "--drift-npz",
        drift_npz,
        "--preset",
        str(args.preset),
        "--seeds",
        str(args.seeds),
        "--episodes",
        str(int(args.episodes)),
        "--dynamics-model",
        str(args.dynamics_model),
        "--constraint-mode",
        str(args.constraint_mode),
        "--bathy-mode",
        str(args.bathy_mode),
        "--seafloor-clearance-m",
        str(float(args.seafloor_clearance_m)),
        "--current-gain",
        str(float(args.current_gain)),
        "--rec-step-stride",
        str(int(max(1, int(args.rec_step_stride)))),
        "--validate",
    ]

    jobs: list[dict[str, object]] = []

    # Heuristic baseline.
    jobs.append(
        {
            "name": f"paper_v1_heuristic_mh_cg{float(args.current_gain):.2f}",
            "cmd": [*base, "--out-dir", str(out_root / "heuristic")],
        }
    )

    # BC baseline (optional).
    if str(args.bc_weights_npz).strip():
        jobs.append(
            {
                "name": f"paper_v1_mlp_bc_mh_cg{float(args.current_gain):.2f}",
                "cmd": [
                    *base,
                    "--controller-override",
                    "mlp_bc",
                    "--bc-weights-npz",
                    str(Path(args.bc_weights_npz).expanduser().resolve()),
                    "--out-dir",
                    str(out_root / "mlp_bc"),
                ],
            }
        )

    if not bool(args.no_llm):
        models = [
            ("chatglm3_6b", "/data/shared/user2/models/ChatGLM3-6B"),
            ("glm4_9b", "/data/shared/user2/models/GLM-4-9B-Chat"),
            ("llama2_7b", "/data/shared/user2/models/LLaMA-2-7B-Chat"),
            ("llama3_8b", "/data/shared/user2/models/LLaMA-3-8B-Instruct"),
            ("mistral7b", "/data/shared/user2/models/Mistral-7B-Instruct-v0.3"),
            ("olmo7b", "/data/shared/user2/models/OLMo-7B-Instruct"),
            ("qwen2_7b", "/data/shared/user2/models/Qwen2-7B-Instruct"),
            ("qwen2p5_7b", "/data/shared/user2/models/Qwen2.5-7B-Instruct"),
            ("qwen2p5_14b", "/data/shared/user2/models/Qwen2.5-14B-Instruct"),
        ]
        for mid, mpath in models:
            cache_dir = ""
            if llm_cache_root:
                cache_dir = str(Path(llm_cache_root) / f"paper_v1_{mid}")
            cmd = [
                *base,
                "--controller-override",
                "llm_planner",
                "--llm-model-path",
                str(mpath),
            ]
            if cache_dir:
                cmd += ["--llm-cache-dir", cache_dir]
            cmd += [
                "--llm-call-stride-steps",
                str(int(args.llm_call_stride_steps)),
                "--llm-max-new-tokens",
                str(int(args.llm_max_new_tokens)),
                "--out-dir",
                str(out_root / f"llm_{mid}"),
            ]
            jobs.append({"name": f"paper_v1_llm_{mid}_mh_cg{float(args.current_gain):.2f}", "cmd": cmd})

    with jobs_path.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(str(jobs_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
