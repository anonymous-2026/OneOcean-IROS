from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LLMPlannerConfig:
    model_path: str
    cache_dir: str
    call_stride_steps: int = 30
    max_new_tokens: int = 192


class LLMPlanner:
    def __init__(self, cfg: LLMPlannerConfig) -> None:
        self.cfg = cfg
        self.cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve() if str(cfg.cache_dir).strip() else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded (only when first used).
        self._tok = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._tok is not None and self._model is not None:
            return
        mp = str(self.cfg.model_path).strip()
        if not mp:
            raise ValueError("LLMPlanner requires a non-empty model_path.")
        cached = _GLOBAL_MODEL_CACHE.get(mp)
        if cached is not None:
            self._tok, self._model = cached
            return
        # Import torch/transformers only when LLM planning is enabled.
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch

        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mp,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto",
        )
        model.eval()
        self._tok = tok
        self._model = model
        _GLOBAL_MODEL_CACHE[mp] = (tok, model)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def _cached_get(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cached_put(self, key: str, obj: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        p = self.cache_dir / f"{key}.json"
        try:
            # Best-effort atomic write to avoid corrupting the cache under multi-process runs.
            tmp = self.cache_dir / f"{key}.tmp.{os.getpid()}"
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            return

    def plan_cleanup_assignment(
        self,
        *,
        task_kind: str,
        step_index: int,
        positions_xyz: np.ndarray,
        sources_xyz: np.ndarray,
        done_mask: np.ndarray,
        n_agents: int,
    ) -> list[int] | None:
        """Return a list length N: each entry is source index or -1.

        Deterministic (do_sample=False), schema-validated, and cached.
        """
        pos = np.asarray(positions_xyz, dtype=np.float64).reshape(n_agents, 3)
        src = np.asarray(sources_xyz, dtype=np.float64).reshape(-1, 3)
        done = np.asarray(done_mask, dtype=bool).reshape(-1)

        payload = {
            "task": str(task_kind),
            "step_index": int(step_index),
            "n_agents": int(n_agents),
            "sources_xyz": np.round(src, 2).tolist(),
            "done": done.astype(int).tolist(),
            "agents_xyz": np.round(pos, 2).tolist(),
        }
        key = self._cache_key({"model": str(self.cfg.model_path), **payload})
        cached = self._cached_get(key)
        if isinstance(cached, dict) and isinstance(cached.get("assign"), list):
            return self._validate_assign(cached.get("assign"), n_agents=n_agents, n_sources=int(src.shape[0]), done=done)

        self._ensure_model()
        assert self._tok is not None and self._model is not None

        sys_txt = (
            "You are a planner for multi-agent underwater cleanup. "
            "Your job is to assign each agent to a cleanup source index.\n"
            "Return ONLY valid JSON with keys: assign (list[int]), rationale (string).\n"
            "Rules: length(assign)=N; each value is -1 or an integer in [0,S-1]; do not assign DONE sources."
        )
        user_txt = (
            f"Task={task_kind}\n"
            f"N={n_agents} agents\n"
            f"S={int(src.shape[0])} sources (xyz): {payload['sources_xyz']}\n"
            f"DONE mask: {payload['done']}\n"
            f"Agents (xyz): {payload['agents_xyz']}\n"
            "Output JSON now."
        )

        tok = self._tok
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
        else:
            text = sys_txt + "\n\n" + user_txt + "\n\nJSON:"

        import torch

        inputs = tok(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=False,
                top_p=1.0,
                top_k=0,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)

        parsed = _extract_json(decoded)
        if not isinstance(parsed, dict):
            self._cached_put(key, {"error": "parse_failed", "raw": decoded})
            return None
        assign = parsed.get("assign", None)
        valid = self._validate_assign(assign, n_agents=n_agents, n_sources=int(src.shape[0]), done=done)
        if valid is None:
            self._cached_put(key, {"error": "schema_failed", "raw": decoded, "parsed": parsed})
            return None
        out_obj = {"assign": valid, "rationale": str(parsed.get("rationale", ""))}
        self._cached_put(key, out_obj)
        return valid

    def plan_waypoint_assignment(
        self,
        *,
        task_kind: str,
        step_index: int,
        positions_xyz: np.ndarray,
        waypoints_xyz: np.ndarray,
        n_agents: int,
        detected_mask: np.ndarray | None = None,
    ) -> list[int] | None:
        """Assign each agent a waypoint index along a path (high-level planning)."""
        pos = np.asarray(positions_xyz, dtype=np.float64).reshape(n_agents, 3)
        wps = np.asarray(waypoints_xyz, dtype=np.float64).reshape(-1, 3)
        if wps.shape[0] < 2:
            return None

        det = None
        if detected_mask is not None:
            try:
                det = np.asarray(detected_mask, dtype=bool).reshape(-1)
            except Exception:
                det = None

        # Downsample the waypoint list in the prompt to keep it compact/deterministic.
        stride = int(max(1, math.ceil(float(wps.shape[0]) / 24.0)))
        payload = {
            "task": str(task_kind),
            "step_index": int(step_index),
            "n_agents": int(n_agents),
            "waypoints_preview_xyz": np.round(wps[::stride], 2).tolist(),
            "agents_xyz": np.round(pos, 2).tolist(),
            "detected_mask": det.astype(int).tolist() if det is not None else None,
        }
        key = self._cache_key({"model": str(self.cfg.model_path), "kind": "waypoint_assignment", **payload})
        cached = self._cached_get(key)
        if isinstance(cached, dict) and isinstance(cached.get("assign_wp"), list):
            return self._validate_wp_assign(cached.get("assign_wp"), n_agents=n_agents, n_wp=int(wps.shape[0]))

        self._ensure_model()
        assert self._tok is not None and self._model is not None

        sys_txt = (
            "You are a planner for multi-agent underwater search.\n"
            "Assign each agent a waypoint index along a path so agents spread out to cover the path.\n"
            "Return ONLY valid JSON with keys: assign_wp (list[int]), rationale (string).\n"
            "Rules: length(assign_wp)=N; each value is an integer in [0,K-1]."
        )
        user_txt = (
            f"Task={task_kind}\n"
            f"N={n_agents} agents\n"
            f"K={int(wps.shape[0])} waypoints (preview): {payload['waypoints_preview_xyz']}\n"
            f"Agents (xyz): {payload['agents_xyz']}\n"
            f"Detected mask (optional): {payload['detected_mask']}\n"
            "Output JSON now."
        )

        tok = self._tok
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
        else:
            text = sys_txt + "\n\n" + user_txt + "\n\nJSON:"

        import torch

        inputs = tok(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=False,
                top_p=1.0,
                top_k=0,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)

        parsed = _extract_json(decoded)
        if not isinstance(parsed, dict):
            self._cached_put(key, {"error": "parse_failed", "raw": decoded})
            return None
        assign = parsed.get("assign_wp", None)
        valid = self._validate_wp_assign(assign, n_agents=n_agents, n_wp=int(wps.shape[0]))
        if valid is None:
            self._cached_put(key, {"error": "schema_failed", "raw": decoded, "parsed": parsed})
            return None
        out_obj = {"assign_wp": valid, "rationale": str(parsed.get("rationale", ""))}
        self._cached_put(key, out_obj)
        return valid

    @staticmethod
    def _validate_assign(assign_any: Any, *, n_agents: int, n_sources: int, done: np.ndarray) -> list[int] | None:
        if not isinstance(assign_any, list) or len(assign_any) != int(n_agents):
            return None
        out: list[int] = []
        for a in assign_any:
            try:
                ai = int(a)
            except Exception:
                return None
            if ai == -1:
                out.append(-1)
                continue
            if ai < 0 or ai >= int(n_sources):
                return None
            if bool(done[ai]):
                return None
            out.append(ai)
        return out

    @staticmethod
    def _validate_wp_assign(assign_any: Any, *, n_agents: int, n_wp: int) -> list[int] | None:
        if not isinstance(assign_any, list) or len(assign_any) != int(n_agents):
            return None
        out: list[int] = []
        for a in assign_any:
            try:
                ii = int(a)
            except Exception:
                return None
            if ii < 0 or ii >= int(n_wp):
                return None
            out.append(ii)
        return out


def _extract_json(text: str) -> Any:
    s = str(text)
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    frag = s[i : j + 1]
    try:
        return json.loads(frag)
    except Exception:
        return None


_GLOBAL_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
