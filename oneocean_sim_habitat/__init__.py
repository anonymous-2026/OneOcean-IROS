"""S2 Habitat-Lab visual track for OneOcean."""

__all__ = [
    "RunConfig",
    "run_habitat_ocean_proxy",
    "BatchConfig",
    "run_batch_regression",
    "build_media_package",
]


def __getattr__(name: str):  # type: ignore[override]
    # Avoid importing heavy optional deps (e.g., cv2) at package import time.
    if name in ("BatchConfig", "run_batch_regression"):
        from .batch_regression import BatchConfig, run_batch_regression

        return {"BatchConfig": BatchConfig, "run_batch_regression": run_batch_regression}[name]
    if name == "build_media_package":
        from .build_media_package import build_media_package

        return build_media_package
    if name in ("RunConfig", "run_habitat_ocean_proxy"):
        from .runner import RunConfig, run_habitat_ocean_proxy

        return {"RunConfig": RunConfig, "run_habitat_ocean_proxy": run_habitat_ocean_proxy}[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
