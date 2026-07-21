from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


IssueLevel = Literal["warning", "error"]


@dataclass(frozen=True)
class ValidationIssue:
    level: IssueLevel
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentProductReport:
    path: str
    status: str = "PASS"
    canonical_mapping: dict[str, str] = field(default_factory=dict)
    dimensions: dict[str, int] = field(default_factory=dict)
    coordinates: dict[str, dict[str, Any]] = field(default_factory=dict)
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_attributes: dict[str, Any] = field(default_factory=dict)
    roundtrip: dict[str, Any] | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def finalize(self) -> None:
        if any(issue.level == "error" for issue in self.issues):
            self.status = "FAIL"
        elif self.issues:
            self.status = "PASS_WITH_WARNINGS"
        else:
            self.status = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "canonical_mapping": self.canonical_mapping,
            "dimensions": self.dimensions,
            "coordinates": self.coordinates,
            "variables": self.variables,
            "global_attributes": self.global_attributes,
            "roundtrip": self.roundtrip,
            "issues": [asdict(issue) for issue in self.issues],
        }


COORDINATE_ALIASES = {
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
    "time": ("time",),
    "depth": ("depth",),
}

VARIABLE_ALIASES = {
    "elevation": ("elevation", "bathymetry", "terrain_elevation"),
    "land_mask": ("land_mask", "invalid_mask", "mask"),
}

CURRENT_PAIRS = (
    ("utotal", "vtotal"),
    ("uo", "vo"),
    ("water_u", "water_v"),
    ("u", "v"),
)

CURRENT_UNITS = {"m/s", "m s-1", "m s^-1", "m s**-1", "meter second-1", "meters per second"}
LATITUDE_UNITS = {"degrees_north", "degree_north", "degrees north"}
LONGITUDE_UNITS = {"degrees_east", "degree_east", "degrees east"}
DEPTH_UNITS = {"m", "meter", "meters", "metre", "metres"}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _resolve_name(available: set[str], aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in available), None)


def _normalized_units(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("/", "/").split())


def _issue(
    report: EnvironmentProductReport,
    level: IssueLevel,
    code: str,
    message: str,
    **details: Any,
) -> None:
    report.issues.append(ValidationIssue(level=level, code=code, message=message, details=_json_value(details)))


def _sample_values(data_array: Any, max_points: int) -> np.ndarray:
    if not data_array.dims:
        return np.asarray(data_array.values)
    points_per_dim = max(1, int(round(max_points ** (1.0 / max(1, data_array.ndim)))))
    indexers: dict[str, slice] = {}
    for dim, size in data_array.sizes.items():
        stride = max(1, int(math.ceil(int(size) / points_per_dim)))
        indexers[str(dim)] = slice(0, int(size), stride)
    return np.asarray(data_array.isel(indexers).values)


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return {"sample_count": int(array.size)}
    finite = np.isfinite(array)
    finite_values = array[finite]
    summary: dict[str, Any] = {
        "sample_count": int(array.size),
        "finite_count": int(finite_values.size),
        "nan_or_inf_fraction": float(1.0 - finite_values.size / max(1, array.size)),
    }
    if finite_values.size:
        summary.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
            }
        )
    return summary


def _max_abs_error(first: np.ndarray, second: np.ndarray) -> float | None:
    differences = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    finite = differences[np.isfinite(differences)]
    return float(np.max(finite)) if finite.size else None


def _coordinate_summary(data_array: Any, max_points: int) -> dict[str, Any]:
    values = _sample_values(data_array, max_points=max_points).reshape(-1)
    summary = {
        "source_name": str(data_array.name),
        "dims": [str(dim) for dim in data_array.dims],
        "size": int(data_array.size),
        "dtype": str(data_array.dtype),
        "units": _json_value(data_array.attrs.get("units")),
        **_numeric_summary(values),
    }
    if values.size > 1 and np.issubdtype(values.dtype, np.number):
        diffs = np.diff(values.astype(np.float64))
        summary["strictly_increasing"] = bool(np.all(diffs > 0.0))
        summary["strictly_decreasing"] = bool(np.all(diffs < 0.0))
    return summary


def _variable_summary(data_array: Any, max_points: int) -> dict[str, Any]:
    values = _sample_values(data_array, max_points=max_points)
    return {
        "source_name": str(data_array.name),
        "dims": [str(dim) for dim in data_array.dims],
        "shape": [int(size) for size in data_array.shape],
        "dtype": str(data_array.dtype),
        "units": _json_value(data_array.attrs.get("units")),
        "long_name": _json_value(data_array.attrs.get("long_name")),
        "standard_name": _json_value(data_array.attrs.get("standard_name")),
        **_numeric_summary(values),
    }


def _validate_coordinate(
    report: EnvironmentProductReport,
    canonical_name: str,
    data_array: Any,
    accepted_units: set[str] | None,
    strict_units: bool,
    max_points: int,
) -> None:
    summary = _coordinate_summary(data_array, max_points=max_points)
    report.coordinates[canonical_name] = summary
    values = np.asarray(data_array.values).reshape(-1)
    if data_array.ndim != 1:
        _issue(report, "error", "coordinate_not_1d", f"{canonical_name} must be one-dimensional", shape=list(data_array.shape))
        return
    if not values.size or not np.all(np.isfinite(values)):
        _issue(report, "error", "coordinate_nonfinite", f"{canonical_name} contains NaN/Inf or is empty")
    if values.size > 1 and not np.all(np.diff(values.astype(np.float64)) > 0.0):
        _issue(report, "error", "coordinate_not_ascending", f"{canonical_name} must be strictly increasing")
    if canonical_name == "latitude" and values.size and (float(np.min(values)) < -90.0 or float(np.max(values)) > 90.0):
        _issue(report, "error", "latitude_out_of_range", "latitude must be within [-90, 90]")
    if canonical_name == "longitude" and values.size and (float(np.min(values)) < -180.0 or float(np.max(values)) > 360.0):
        _issue(report, "error", "longitude_out_of_range", "longitude must be within [-180, 360]")
    if canonical_name == "depth" and values.size and float(np.min(values)) < 0.0:
        _issue(report, "warning", "negative_depth", "depth contains negative values; OneOcean expects positive-down meters")
    if accepted_units is None:
        return
    units = _normalized_units(data_array.attrs.get("units"))
    if not units:
        _issue(report, "warning", "missing_coordinate_units", f"{canonical_name} has no units attribute")
    elif units not in accepted_units:
        _issue(
            report,
            "error" if strict_units else "warning",
            "unexpected_coordinate_units",
            f"{canonical_name} uses unexpected units",
            units=units,
            accepted=sorted(accepted_units),
        )


def _resolve_current_pair(
    data_variables: set[str],
    requested_u: str | None,
    requested_v: str | None,
) -> tuple[str | None, str | None]:
    if requested_u or requested_v:
        if not requested_u or not requested_v:
            return None, None
        return (requested_u, requested_v) if requested_u in data_variables and requested_v in data_variables else (None, None)
    return next(((u_name, v_name) for u_name, v_name in CURRENT_PAIRS if u_name in data_variables and v_name in data_variables), (None, None))


def _validate_current_units(
    report: EnvironmentProductReport,
    canonical_name: str,
    data_array: Any,
    strict_units: bool,
) -> None:
    units = _normalized_units(data_array.attrs.get("units"))
    if not units:
        _issue(report, "warning", "missing_current_units", f"{canonical_name} has no units attribute; expected m/s")
    elif units not in CURRENT_UNITS:
        _issue(
            report,
            "error" if strict_units else "warning",
            "unexpected_current_units",
            f"{canonical_name} uses unexpected units",
            units=units,
            accepted=sorted(CURRENT_UNITS),
        )


def _extract_current_slice(data_array: Any, latitude_dim: str, longitude_dim: str, time_index: int, depth_index: int) -> np.ndarray:
    indexers: dict[str, int] = {}
    for dim in data_array.dims:
        if dim in {latitude_dim, longitude_dim}:
            continue
        if dim == "time":
            indexers[dim] = int(np.clip(time_index, 0, int(data_array.sizes[dim]) - 1))
        elif dim == "depth":
            indexers[dim] = int(np.clip(depth_index, 0, int(data_array.sizes[dim]) - 1))
        else:
            indexers[dim] = 0
    sliced = data_array.isel(indexers).transpose(latitude_dim, longitude_dim)
    return np.asarray(sliced.values, dtype=np.float64)


def _validate_roundtrip(
    report: EnvironmentProductReport,
    dataset: Any,
    npz_path: Path,
    latitude_name: str,
    longitude_name: str,
    u_name: str,
    v_name: str,
    time_index: int,
    depth_index: int,
    atol: float,
) -> None:
    payload: dict[str, Any] = {"npz": str(npz_path), "atol": float(atol)}
    if not npz_path.exists():
        _issue(report, "error", "missing_roundtrip_npz", "drift NPZ does not exist", path=str(npz_path))
        report.roundtrip = payload
        return
    try:
        with np.load(npz_path, allow_pickle=False) as cache:
            missing = sorted({"latitude", "longitude", "u", "v"} - set(cache.files))
            if missing:
                _issue(report, "error", "roundtrip_npz_schema", "drift NPZ is missing required arrays", missing=missing)
                report.roundtrip = payload
                return
            lat_cache = np.asarray(cache["latitude"], dtype=np.float64)
            lon_cache = np.asarray(cache["longitude"], dtype=np.float64)
            u_cache = np.asarray(cache["u"], dtype=np.float64)
            v_cache = np.asarray(cache["v"], dtype=np.float64)
    except Exception as exc:
        _issue(report, "error", "roundtrip_npz_read", "failed to read drift NPZ", error=f"{type(exc).__name__}: {exc}")
        report.roundtrip = payload
        return

    lat_source = np.asarray(dataset[latitude_name].values, dtype=np.float64)
    lon_source = np.asarray(dataset[longitude_name].values, dtype=np.float64)
    u_source = _extract_current_slice(dataset[u_name], latitude_name, longitude_name, time_index, depth_index)
    v_source = _extract_current_slice(dataset[v_name], latitude_name, longitude_name, time_index, depth_index)

    payload.update(
        {
            "time_index": int(time_index),
            "depth_index": int(depth_index),
            "coordinate_match": bool(np.array_equal(lat_source, lat_cache) and np.array_equal(lon_source, lon_cache)),
            "u_max_abs_error": _max_abs_error(u_source, u_cache) if u_source.shape == u_cache.shape else None,
            "v_max_abs_error": _max_abs_error(v_source, v_cache) if v_source.shape == v_cache.shape else None,
        }
    )
    if not payload["coordinate_match"]:
        _issue(report, "error", "roundtrip_coordinate_mismatch", "NetCDF and drift NPZ coordinates differ")
    if u_source.shape != u_cache.shape or v_source.shape != v_cache.shape:
        _issue(
            report,
            "error",
            "roundtrip_shape_mismatch",
            "NetCDF current slice and drift NPZ shapes differ",
            source_u=list(u_source.shape),
            cache_u=list(u_cache.shape),
            source_v=list(v_source.shape),
            cache_v=list(v_cache.shape),
        )
    else:
        if not np.allclose(u_source, u_cache, rtol=0.0, atol=atol, equal_nan=True):
            _issue(report, "error", "roundtrip_u_mismatch", "eastward current differs after NPZ export", max_abs_error=payload["u_max_abs_error"])
        if not np.allclose(v_source, v_cache, rtol=0.0, atol=atol, equal_nan=True):
            _issue(report, "error", "roundtrip_v_mismatch", "northward current differs after NPZ export", max_abs_error=payload["v_max_abs_error"])
    report.roundtrip = payload


def validate_environment_product(
    path: str | Path,
    *,
    u_var: str | None = None,
    v_var: str | None = None,
    strict_units: bool = False,
    max_sample_points: int = 100_000,
    drift_npz: str | Path | None = None,
    time_index: int = 0,
    depth_index: int = 0,
    roundtrip_atol: float = 1e-6,
) -> EnvironmentProductReport:
    product_path = Path(path).expanduser().resolve()
    report = EnvironmentProductReport(path=str(product_path))
    if not product_path.exists():
        _issue(report, "error", "missing_file", "environment product does not exist", path=str(product_path))
        report.finalize()
        return report

    try:
        import xarray as xr
    except ImportError:
        _issue(report, "error", "missing_xarray", "xarray is required; install the repository requirements")
        report.finalize()
        return report

    try:
        dataset = xr.open_dataset(product_path, decode_times=False)
    except Exception as exc:
        _issue(report, "error", "open_failed", "failed to open environment product", error=f"{type(exc).__name__}: {exc}")
        report.finalize()
        return report

    try:
        report.dimensions = {str(name): int(size) for name, size in dataset.sizes.items()}
        available = set(dataset.variables)
        data_variables = set(dataset.data_vars)

        for canonical, aliases in COORDINATE_ALIASES.items():
            resolved = _resolve_name(available, aliases)
            if resolved is not None:
                report.canonical_mapping[canonical] = resolved

        for required in ("latitude", "longitude"):
            if required not in report.canonical_mapping:
                _issue(report, "error", "missing_coordinate", f"missing required {required} coordinate", aliases=list(COORDINATE_ALIASES[required]))

        if "latitude" in report.canonical_mapping:
            _validate_coordinate(
                report,
                "latitude",
                dataset[report.canonical_mapping["latitude"]],
                LATITUDE_UNITS,
                strict_units,
                max_sample_points,
            )
        if "longitude" in report.canonical_mapping:
            _validate_coordinate(
                report,
                "longitude",
                dataset[report.canonical_mapping["longitude"]],
                LONGITUDE_UNITS,
                strict_units,
                max_sample_points,
            )
        if "time" in report.canonical_mapping:
            _validate_coordinate(report, "time", dataset[report.canonical_mapping["time"]], None, strict_units, max_sample_points)
            if not dataset[report.canonical_mapping["time"]].attrs.get("units"):
                _issue(report, "warning", "missing_time_units", "time has no CF-style units attribute")
        if "depth" in report.canonical_mapping:
            _validate_coordinate(
                report,
                "depth",
                dataset[report.canonical_mapping["depth"]],
                DEPTH_UNITS,
                strict_units,
                max_sample_points,
            )

        for canonical, aliases in VARIABLE_ALIASES.items():
            resolved = _resolve_name(data_variables, aliases)
            if resolved is None:
                _issue(report, "error", "missing_variable", f"missing required {canonical} variable", aliases=list(aliases))
            else:
                report.canonical_mapping[canonical] = resolved
                report.variables[canonical] = _variable_summary(dataset[resolved], max_sample_points)

        selected_u, selected_v = _resolve_current_pair(data_variables, u_var, v_var)
        if selected_u is None or selected_v is None:
            _issue(
                report,
                "error",
                "missing_current_pair",
                "no supported eastward/northward current pair was found",
                requested=[u_var, v_var],
                supported=[list(pair) for pair in CURRENT_PAIRS],
            )
        else:
            report.canonical_mapping["eastward_current"] = selected_u
            report.canonical_mapping["northward_current"] = selected_v
            u_data = dataset[selected_u]
            v_data = dataset[selected_v]
            report.variables["eastward_current"] = _variable_summary(u_data, max_sample_points)
            report.variables["northward_current"] = _variable_summary(v_data, max_sample_points)
            if u_data.dims != v_data.dims or u_data.shape != v_data.shape:
                _issue(report, "error", "current_shape_mismatch", "eastward and northward currents must have identical dimensions and shape")
            latitude_dim = report.canonical_mapping.get("latitude")
            longitude_dim = report.canonical_mapping.get("longitude")
            for dim_name in (latitude_dim, longitude_dim):
                if dim_name is not None and dim_name not in u_data.dims:
                    _issue(report, "error", "current_missing_spatial_dim", "current variables must include latitude and longitude dimensions", missing_dim=dim_name)
            _validate_current_units(report, "eastward_current", u_data, strict_units)
            _validate_current_units(report, "northward_current", v_data, strict_units)

        if "elevation" in report.canonical_mapping:
            elevation = dataset[report.canonical_mapping["elevation"]]
            latitude_dim = report.canonical_mapping.get("latitude")
            longitude_dim = report.canonical_mapping.get("longitude")
            expected_dims = tuple(dim for dim in (latitude_dim, longitude_dim) if dim is not None)
            if tuple(elevation.dims) != expected_dims:
                _issue(report, "error", "elevation_dims", "elevation must use (latitude, longitude) dimensions", actual=list(elevation.dims), expected=list(expected_dims))
            if not elevation.attrs.get("units"):
                _issue(report, "warning", "missing_elevation_units", "elevation has no units attribute; expected meters relative to sea surface")

        if "land_mask" in report.canonical_mapping:
            mask = dataset[report.canonical_mapping["land_mask"]]
            latitude_dim = report.canonical_mapping.get("latitude")
            longitude_dim = report.canonical_mapping.get("longitude")
            expected_dims = tuple(dim for dim in (latitude_dim, longitude_dim) if dim is not None)
            if tuple(mask.dims) != expected_dims:
                _issue(report, "error", "land_mask_dims", "land_mask must use (latitude, longitude) dimensions", actual=list(mask.dims), expected=list(expected_dims))
            mask_sample = _sample_values(mask, max_sample_points)
            finite_mask = mask_sample[np.isfinite(mask_sample)]
            if finite_mask.size and (float(np.min(finite_mask)) < 0.0 or float(np.max(finite_mask)) > 1.0):
                _issue(report, "error", "land_mask_range", "land_mask values must be in [0, 1]")
            if not mask.attrs.get("flag_meanings"):
                _issue(report, "warning", "missing_mask_semantics", "land_mask should define flag_meanings for valid and invalid terrain")

        global_attr_names = (
            "Conventions",
            "source",
            "generated_at_utc",
            "basic_dataset_id",
            "include_tides",
            "depth_request_mode",
            "requested_depth_values_m",
            "actual_depth_values_m",
            "minimum_depth_m",
            "maximum_depth_m",
            "interpolation_method",
        )
        report.global_attributes = {name: _json_value(dataset.attrs[name]) for name in global_attr_names if name in dataset.attrs}
        for recommended in ("source", "generated_at_utc", "interpolation_method"):
            if recommended not in dataset.attrs:
                _issue(report, "warning", "missing_provenance", f"missing recommended provenance attribute: {recommended}")

        if drift_npz is not None and selected_u is not None and selected_v is not None:
            latitude_name = report.canonical_mapping.get("latitude")
            longitude_name = report.canonical_mapping.get("longitude")
            if latitude_name is not None and longitude_name is not None:
                _validate_roundtrip(
                    report,
                    dataset,
                    Path(drift_npz).expanduser().resolve(),
                    latitude_name,
                    longitude_name,
                    selected_u,
                    selected_v,
                    time_index,
                    depth_index,
                    roundtrip_atol,
                )
    except Exception as exc:
        _issue(report, "error", "validation_failed", "unexpected validation failure", error=f"{type(exc).__name__}: {exc}")
    finally:
        dataset.close()

    report.finalize()
    return report


def write_environment_report(report: EnvironmentProductReport, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path
