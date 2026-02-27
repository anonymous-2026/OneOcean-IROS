from __future__ import annotations

from pathlib import Path


def test_h3_ocpnet_plume_sink_reduces_mass(tmp_path: Path) -> None:
    from tracks.h3_oceangym.ocpnet_plume import OCPNetCfg, OCPNetPlume

    plume = OCPNetPlume(cfg=OCPNetCfg(time_step_s=0.05, grid_resolution=(16, 16, 8)), work_dir=tmp_path, world_center_xyz=(0.0, 0.0, 0.0))
    plume.set_source_world((0.0, 0.0, 0.0))

    # Step a little to build up some concentration.
    for _ in range(5):
        plume.step(u_mps=0.05, v_mps=0.02)

    m0 = plume.mass_total()
    removed = plume.apply_sink_at_world((0.0, 0.0, 0.0), sink_radius_m=10.0, sink_strength=0.5)
    m1 = plume.mass_total()

    assert m1 <= m0
    assert removed >= 0.0

