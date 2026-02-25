from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int32


def _ns(root: ET.Element) -> dict[str, str]:
    uri = root.tag.split("}")[0].strip("{")
    return {"c": uri}


def _parse_floats(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros((0,), dtype=np.float64)
    return np.fromstring(text.strip(), sep=" ", dtype=np.float64)


def _parse_ints(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros((0,), dtype=np.int64)
    return np.fromstring(text.strip(), sep=" ", dtype=np.int64)


def _parse_matrix_row_major(elem: ET.Element) -> np.ndarray:
    vals = _parse_floats(elem.text)
    if vals.size != 16:
        raise ValueError("Expected 16 values in <matrix>")
    return vals.reshape(4, 4).astype(np.float64)


def _transform_points(points: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([points.astype(np.float64), ones], axis=1)
    out = (hom @ mat4.T)[:, :3]
    return out.astype(np.float64)


def _collect_sources(mesh_elem: ET.Element, ns: dict[str, str]) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}
    for source in mesh_elem.findall("c:source", ns):
        sid = source.attrib.get("id")
        if not sid:
            continue
        float_array = source.find("c:float_array", ns)
        floats = _parse_floats(float_array.text if float_array is not None else None)
        accessor = source.find("c:technique_common/c:accessor", ns)
        if accessor is None:
            sources[sid] = floats.astype(np.float64)
            continue
        stride = int(accessor.attrib.get("stride", "1"))
        offset = int(accessor.attrib.get("offset", "0"))
        count = int(accessor.attrib.get("count", str(max(0, (len(floats) - offset) // max(1, stride)))))
        rows: list[np.ndarray] = []
        for i in range(count):
            start = offset + i * stride
            end = start + stride
            rows.append(floats[start:end])
        data = np.asarray(rows, dtype=np.float64)
        sources[sid] = data
    return sources


def _geometry_mesh(geometry: ET.Element, ns: dict[str, str]) -> TriangleMesh:
    mesh = geometry.find("c:mesh", ns)
    if mesh is None:
        return TriangleMesh(vertices=np.zeros((0, 3), dtype=np.float64), faces=np.zeros((0, 3), dtype=np.int32))

    sources = _collect_sources(mesh, ns)

    # Map <vertices id> -> positions array (Nx3).
    vertices_map: dict[str, np.ndarray] = {}
    for v in mesh.findall("c:vertices", ns):
        vid = v.attrib.get("id")
        if not vid:
            continue
        pos_input = None
        for inp in v.findall("c:input", ns):
            if inp.attrib.get("semantic") == "POSITION":
                pos_input = inp
                break
        if pos_input is None:
            continue
        src = pos_input.attrib.get("source", "")
        if src.startswith("#"):
            src = src[1:]
        arr = sources.get(src)
        if arr is None:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        vertices_map[vid] = np.asarray(arr[:, :3], dtype=np.float64)

    geom_vertices: np.ndarray | None = None
    geom_faces: list[np.ndarray] = []

    for tris in mesh.findall("c:triangles", ns):
        count = int(tris.attrib.get("count", "0"))
        if count <= 0:
            continue

        inputs = tris.findall("c:input", ns)
        if not inputs:
            continue

        max_offset = 0
        vertex_offset = None
        vertex_source = None
        for inp in inputs:
            off = int(inp.attrib.get("offset", "0"))
            max_offset = max(max_offset, off)
            if inp.attrib.get("semantic") == "VERTEX":
                vertex_offset = off
                vertex_source = inp.attrib.get("source", "")
        if vertex_offset is None or not vertex_source:
            continue
        if vertex_source.startswith("#"):
            vertex_source = vertex_source[1:]

        positions = vertices_map.get(vertex_source)
        if positions is None:
            continue

        p = tris.find("c:p", ns)
        if p is None:
            continue
        idx = _parse_ints(p.text)
        stride = max_offset + 1
        expected = count * 3 * stride
        if idx.size < expected:
            # Some files omit trailing whitespace; tolerate only exact length.
            continue

        if geom_vertices is None:
            geom_vertices = np.asarray(positions[:, :3], dtype=np.float64)
            base = 0
        else:
            base = 0
        faces = np.zeros((count, 3), dtype=np.int32)
        for ti in range(count):
            for corner in range(3):
                faces[ti, corner] = int(base + int(idx[(ti * 3 + corner) * stride + vertex_offset]))
        geom_faces.append(faces)

    if geom_vertices is None or not geom_faces:
        return TriangleMesh(vertices=np.zeros((0, 3), dtype=np.float64), faces=np.zeros((0, 3), dtype=np.int32))

    faces_cat = np.concatenate(geom_faces, axis=0).astype(np.int32)
    return TriangleMesh(vertices=geom_vertices.astype(np.float64), faces=faces_cat)


def _walk_nodes(
    node: ET.Element,
    ns: dict[str, str],
    parent_mat: np.ndarray,
    out_instances: list[tuple[str, np.ndarray]],
) -> None:
    mat = parent_mat
    matrix = node.find("c:matrix", ns)
    if matrix is not None and (matrix.text or "").strip():
        mat = parent_mat @ _parse_matrix_row_major(matrix)

    for inst in node.findall("c:instance_geometry", ns):
        url = inst.attrib.get("url", "")
        if url.startswith("#"):
            url = url[1:]
        if url:
            out_instances.append((url, mat))

    for child in node.findall("c:node", ns):
        _walk_nodes(child, ns, mat, out_instances)


def load_collada_triangles(path: Path) -> TriangleMesh:
    root = ET.parse(path).getroot()
    ns = _ns(root)

    # Global units.
    unit = root.find("c:asset/c:unit", ns)
    meter = float(unit.attrib.get("meter", "1")) if unit is not None else 1.0

    # Build per-geometry triangle meshes.
    geom_mesh: dict[str, TriangleMesh] = {}
    for geom in root.findall("c:library_geometries/c:geometry", ns):
        gid = geom.attrib.get("id")
        if not gid:
            continue
        geom_mesh[gid] = _geometry_mesh(geom, ns)

    # Instance geometries from the visual scene with node transforms.
    instances: list[tuple[str, np.ndarray]] = []
    visual_scene = root.find("c:library_visual_scenes/c:visual_scene", ns)
    if visual_scene is not None:
        for node in visual_scene.findall("c:node", ns):
            _walk_nodes(node, ns, np.eye(4, dtype=np.float64), instances)

    meshes: list[TriangleMesh] = []
    if instances:
        for gid, mat4 in instances:
            m = geom_mesh.get(gid)
            if m is None or m.vertices.size == 0 or m.faces.size == 0:
                continue
            v = _transform_points(m.vertices * meter, mat4)
            meshes.append(TriangleMesh(vertices=v, faces=m.faces.copy()))
    else:
        for m in geom_mesh.values():
            if m.vertices.size == 0 or m.faces.size == 0:
                continue
            meshes.append(TriangleMesh(vertices=m.vertices * meter, faces=m.faces.copy()))

    if not meshes:
        return TriangleMesh(vertices=np.zeros((0, 3), dtype=np.float64), faces=np.zeros((0, 3), dtype=np.int32))

    # Concatenate with face reindexing.
    v_all: list[np.ndarray] = []
    f_all: list[np.ndarray] = []
    offset = 0
    for m in meshes:
        v_all.append(m.vertices.astype(np.float64))
        f_all.append((m.faces.astype(np.int32) + int(offset)).astype(np.int32))
        offset += int(m.vertices.shape[0])

    vertices = np.concatenate(v_all, axis=0).astype(np.float64)
    faces = np.concatenate(f_all, axis=0).astype(np.int32)
    return TriangleMesh(vertices=vertices, faces=faces)


def mesh_to_obj(
    mesh: TriangleMesh,
    *,
    output_obj: Path,
    center: bool = True,
    max_faces: int | None = 4000,
) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if center and vertices.size:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center_pt = 0.5 * (vmin + vmax)
        vertices = (vertices - center_pt[None, :]).astype(np.float64)

    if max_faces is not None and int(max_faces) > 0 and faces.shape[0] > int(max_faces):
        keep = int(max_faces)
        idx = np.linspace(0, faces.shape[0] - 1, num=keep, dtype=np.int64)
        faces = faces[idx].astype(np.int32)
        used = np.unique(faces.reshape(-1))
        remap = {int(old): int(i) for i, old in enumerate(used.tolist())}
        vertices = vertices[used].astype(np.float64)
        faces = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(faces).astype(np.int32)

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_obj.open("w", encoding="utf-8") as file:
        file.write("# Generated by oneocean_sim_s3.external_scenes.collada_loader\n")
        for x, y, z in vertices:
            file.write(f"v {float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        for a, b, c in faces:
            file.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")

    return {
        "output_obj": str(output_obj),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "centered": bool(center),
        "max_faces": None if max_faces is None else int(max_faces),
    }
