"""Deployable V2X geometry features from calibration and time-sync metadata."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GEOV2X_GEOMETRY_DIM = 11


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_vec3(value: Any) -> List[float]:
    if isinstance(value, list) and len(value) == 3:
        out = []
        for item in value:
            if isinstance(item, list):
                out.append(float(item[0]))
            else:
                out.append(float(item))
        return out
    raise ValueError(f"Expected 3-vector, got {value!r}")


def _matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec3(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _transpose3(a: List[List[float]]) -> List[List[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _yaw_from_rotation_xy(rot: List[List[float]]) -> float:
    return math.atan2(rot[1][0], rot[0][0])


@lru_cache(maxsize=1)
def _cooperative_by_vehicle(v2x_root: str) -> Dict[str, Dict[str, Any]]:
    rows = _read_json(Path(v2x_root) / "cooperative" / "data_info.json")
    return {str(row["vehicle_frame"]): row for row in rows}


@lru_cache(maxsize=1)
def _side_info(v2x_root: str, side: str) -> Dict[str, Dict[str, Any]]:
    rows = _read_json(Path(v2x_root) / side / "data_info.json")
    return {str(row["frame_id"]): row for row in rows}


def _vehicle_lidar_to_world(v2x_root: Path, frame: str) -> Tuple[List[List[float]], List[float]]:
    lidar_to_novatel = _read_json(v2x_root / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{frame}.json")
    novatel_to_world = _read_json(v2x_root / "vehicle-side" / "calib" / "novatel_to_world" / f"{frame}.json")
    r_ln = [[float(x) for x in row] for row in lidar_to_novatel["transform"]["rotation"]]
    t_ln = _as_vec3(lidar_to_novatel["transform"]["translation"])
    r_nw = [[float(x) for x in row] for row in novatel_to_world["rotation"]]
    t_nw = _as_vec3(novatel_to_world["translation"])
    r_lw = _matmul3(r_nw, r_ln)
    t_lw = [t_nw[i] + _matvec3(r_nw, t_ln)[i] for i in range(3)]
    return r_lw, t_lw


def _infra_lidar_to_world(v2x_root: Path, frame: str) -> Tuple[List[List[float]], List[float]]:
    payload = _read_json(
        v2x_root / "infrastructure-side" / "calib" / "virtuallidar_to_world" / f"{frame}.json"
    )
    rot = [[float(x) for x in row] for row in payload["rotation"]]
    trans = _as_vec3(payload["translation"])
    return rot, trans


def build_geov2x_geometry(
    token: str,
    v2x_root: str = "data/V2X-Seq-SPD",
) -> Dict[str, Any]:
    """Return normalized ego-frame infrastructure pose and sync features.

    This uses only calibration, timestamps, and cooperative frame mapping. It
    deliberately does not read labels, object boxes, L1/L2/L3/L4 annotations, or
    future trajectory fields.
    """
    root = Path(v2x_root)
    vehicle_frame = str(token)
    coop = _cooperative_by_vehicle(str(root)).get(vehicle_frame)
    if coop is None:
        return {
            "features": [0.0] * GEOV2X_GEOMETRY_DIM,
            "available": False,
            "reason": "missing_cooperative_mapping",
        }
    infra_frame = str(coop["infrastructure_frame"])
    veh_info = _side_info(str(root), "vehicle-side").get(vehicle_frame, {})
    inf_info = _side_info(str(root), "infrastructure-side").get(infra_frame, {})
    try:
        r_vehicle_world, t_vehicle_world = _vehicle_lidar_to_world(root, vehicle_frame)
        r_infra_world, t_infra_world = _infra_lidar_to_world(root, infra_frame)
        r_world_vehicle = _transpose3(r_vehicle_world)
        rel_world = [t_infra_world[i] - t_vehicle_world[i] for i in range(3)]
        rel_vehicle = _matvec3(r_world_vehicle, rel_world)
        dx = float(rel_vehicle[0])
        dy = float(rel_vehicle[1])
        distance = math.sqrt(dx * dx + dy * dy)
        bearing = math.atan2(dy, dx)
        rel_rot = _matmul3(r_world_vehicle, r_infra_world)
        rel_yaw = _yaw_from_rotation_xy(rel_rot)
        veh_ts = float(veh_info.get("image_timestamp", 0.0))
        inf_ts = float(inf_info.get("image_timestamp", 0.0))
        # Timestamps are stored in microseconds.
        time_delta = (inf_ts - veh_ts) / 1_000_000.0
        offset = coop.get("system_error_offset") or {}
        offset_dx = float(offset.get("delta_x", 0.0))
        offset_dy = float(offset.get("delta_y", 0.0))
        features = [
            dx / 100.0,
            dy / 100.0,
            distance / 100.0,
            math.sin(rel_yaw),
            math.cos(rel_yaw),
            math.sin(bearing),
            math.cos(bearing),
            time_delta / 0.5,
            offset_dx / 5.0,
            offset_dy / 5.0,
            1.0,
        ]
        return {
            "features": [float(x) for x in features],
            "available": True,
            "vehicle_frame": vehicle_frame,
            "infrastructure_frame": infra_frame,
            "raw": {
                "dx_m": dx,
                "dy_m": dy,
                "distance_m": distance,
                "bearing_rad": bearing,
                "relative_yaw_rad": rel_yaw,
                "time_delta_s": time_delta,
                "system_error_offset": {"delta_x": offset_dx, "delta_y": offset_dy},
            },
        }
    except Exception as exc:
        return {
            "features": [0.0] * GEOV2X_GEOMETRY_DIM,
            "available": False,
            "vehicle_frame": vehicle_frame,
            "infrastructure_frame": infra_frame,
            "reason": f"{type(exc).__name__}: {exc}",
        }
