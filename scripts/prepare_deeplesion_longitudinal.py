#!/usr/bin/env python3
"""
Build candidate same-lesion longitudinal trajectories from the DLT/DLS annotations.

The Deep Lesion Tracker (DLT) benchmark provides matched lesion pairs from
DeepLesion. This script treats each lesion annotation as a graph node and each
DLT pair as an edge. Connected components with at least N unique studies are
candidate longitudinal trajectories for downstream growth-prediction experiments.

Default outputs:
  outputs_deeplesion_longitudinal/deeplesion_trajectory_summary.csv
  outputs_deeplesion_longitudinal/deeplesion_trajectory_timepoints.csv
  outputs_deeplesion_longitudinal/deeplesion_trajectories_len5.csv
  outputs_deeplesion_longitudinal/deeplesion_longitudinal_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DLT_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/JimmyCai91/DLT/main/data/train.json",
    "valid": "https://raw.githubusercontent.com/JimmyCai91/DLT/main/data/valid.json",
    "test": "https://raw.githubusercontent.com/JimmyCai91/DLT/main/data/test.json",
}


IMAGE_RE = re.compile(
    r"(?P<patient>\d+)_(?P<study>\d+)_(?P<scan>\d+)_(?P<slice_start>\d+)-(?P<slice_end>\d+)"
)


@dataclass(frozen=True)
class ImageId:
    patient_id: str
    study_id: str
    scan_id: str
    slice_start: Optional[int]
    slice_end: Optional[int]
    image_name: str

    @property
    def study_key(self) -> str:
        return f"{self.patient_id}_{self.study_id}_{self.scan_id}"

    @property
    def study_order(self) -> Tuple[int, int, int, int]:
        return (
            safe_int(self.patient_id),
            safe_int(self.study_id),
            safe_int(self.scan_id),
            self.slice_start if self.slice_start is not None else -1,
        )


@dataclass
class LesionNode:
    node_id: str
    image: ImageId
    role: str
    split: str
    center: List[float]
    box: List[float]
    recist_diameter: List[float]
    recist_center: List[float]
    recist_box: List[float]
    spacing: List[float]
    recist_slice: str

    @property
    def long_axis_mm(self) -> Optional[float]:
        return self.recist_diameter[0] if len(self.recist_diameter) >= 1 else None

    @property
    def short_axis_mm(self) -> Optional[float]:
        return self.recist_diameter[1] if len(self.recist_diameter) >= 2 else None

    @property
    def area_proxy_mm2(self) -> Optional[float]:
        if self.long_axis_mm is None or self.short_axis_mm is None:
            return None
        return math.pi * self.long_axis_mm * self.short_axis_mm / 4.0

    @property
    def volume_proxy_mm3(self) -> Optional[float]:
        if self.long_axis_mm is None or self.short_axis_mm is None:
            return None
        return math.pi * self.long_axis_mm * (self.short_axis_mm**2) / 6.0


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def compact_float_list(values: Iterable[float], precision: int) -> str:
    parts = []
    for value in values:
        if isinstance(value, float) and math.isnan(value):
            parts.append("nan")
        else:
            parts.append(f"{float(value):.{precision}f}")
    return ",".join(parts)


def parse_image_name(name: str) -> ImageId:
    base = Path(str(name)).name
    match = IMAGE_RE.search(base)
    if not match:
        stem = base.replace(".nii.gz", "").replace(".png", "")
        fields = stem.split("_")
        patient = fields[0] if len(fields) >= 1 else stem
        study = fields[1] if len(fields) >= 2 else "NA"
        scan = fields[2] if len(fields) >= 3 else "NA"
        return ImageId(patient, study, scan, None, None, base)
    return ImageId(
        patient_id=match.group("patient"),
        study_id=match.group("study"),
        scan_id=match.group("scan"),
        slice_start=int(match.group("slice_start")),
        slice_end=int(match.group("slice_end")),
        image_name=base,
    )


def make_node(pair: Dict[str, Any], split: str, role: str, precision: int) -> LesionNode:
    image = parse_image_name(str(pair.get(role, "")))
    center = as_float_list(pair.get(f"{role} center"))
    box = as_float_list(pair.get(f"{role} box"))
    recist_diameter = as_float_list(pair.get(f"{role} recist diameter"))
    recist_center = as_float_list(pair.get(f"{role} recist center"))
    recist_box = as_float_list(pair.get(f"{role} recist box"))
    spacing = as_float_list(pair.get(f"{role} spacing"))
    recist_slice = str(pair.get(f"{role} recist slice", ""))

    # Use image + lesion geometry. This keeps multiple lesions from the same CT
    # separate while still allowing repeated pair annotations to collapse.
    lesion_fingerprint = "|".join(
        [
            compact_float_list(center, precision),
            compact_float_list(box, precision),
            compact_float_list(recist_diameter, precision),
        ]
    )
    node_id = f"{image.study_key}|{lesion_fingerprint}"
    return LesionNode(
        node_id=node_id,
        image=image,
        role=role,
        split=split,
        center=center,
        box=box,
        recist_diameter=recist_diameter,
        recist_center=recist_center,
        recist_box=recist_box,
        spacing=spacing,
        recist_slice=recist_slice,
    )


def download_missing_json(data_dir: Path, splits: Iterable[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        local_path = data_dir / f"{split}.json"
        if local_path.exists():
            continue
        url = DLT_RAW_URLS.get(split)
        if not url:
            continue
        print(f"Downloading {split}.json from {url}")
        urllib.request.urlretrieve(url, local_path)


def resolve_split_path(data_dir: Path, split: str) -> Optional[Path]:
    candidates = [data_dir / f"{split}.json"]
    if split == "valid":
        candidates.append(data_dir / "val.json")
    if split == "val":
        candidates.append(data_dir / "valid.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def load_pairs(data_dir: Path, splits: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for split in splits:
        path = resolve_split_path(data_dir, split)
        if path is None:
            print(f"WARNING: missing {split}.json under {data_dir}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            split_pairs = []
            for pair_id, item in loaded.items():
                if isinstance(item, dict):
                    item = dict(item)
                    item["_pair_id"] = str(pair_id)
                    split_pairs.append(item)
        elif isinstance(loaded, list):
            split_pairs = []
            for idx, item in enumerate(loaded):
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("_pair_id", str(idx))
                    split_pairs.append(item)
        else:
            raise ValueError(
                f"{path} should contain either a list or a dict of lesion-pair annotations."
            )
        pairs.extend((split, item) for item in split_pairs)
        print(f"Loaded {len(split_pairs)} pairs from {path}")
    return pairs


def build_graph(
    pairs: List[Tuple[str, Dict[str, Any]]],
    precision: int,
) -> Tuple[Dict[str, LesionNode], UnionFind, List[Dict[str, Any]]]:
    nodes: Dict[str, LesionNode] = {}
    uf = UnionFind()
    edges: List[Dict[str, Any]] = []

    for edge_index, (split, pair) in enumerate(pairs):
        source = make_node(pair, split, "source", precision)
        target = make_node(pair, split, "target", precision)
        nodes.setdefault(source.node_id, source)
        nodes.setdefault(target.node_id, target)
        uf.union(source.node_id, target.node_id)
        edges.append(
            {
                "edge_index": edge_index,
                "split": split,
                "source_node_id": source.node_id,
                "target_node_id": target.node_id,
                "source_image": source.image.image_name,
                "target_image": target.image.image_name,
            }
        )
    return nodes, uf, edges


def _has_valid_recist(node: LesionNode) -> bool:
    values = (node.long_axis_mm, node.short_axis_mm)
    return all(value is not None and math.isfinite(value) and value > 0 for value in values)


def select_one_node_per_study(
    nodes: List[LesionNode],
    ambiguity_policy: str = "deterministic",
) -> Tuple[List[LesionNode], int]:
    """Choose one node per study without using lesion outcomes from other visits.

    ``deterministic`` preserves the main cohort: valid RECIST measurements are
    preferred and remaining ties are broken only by the immutable node id.
    ``exclude`` implements the strict sensitivity cohort: a component is
    rejected if any study contains zero or more than one valid RECIST node.
    Neither policy uses past or future lesion size, trajectory smoothness, or
    the eventual prediction target.
    """
    if ambiguity_policy not in {"deterministic", "exclude"}:
        raise ValueError(f"Unknown ambiguity policy: {ambiguity_policy}")
    by_study: Dict[str, List[LesionNode]] = defaultdict(list)
    for node in nodes:
        by_study[node.image.study_key].append(node)

    selected = []
    ambiguous_studies = 0
    for _, group in sorted(by_study.items()):
        valid = sorted((node for node in group if _has_valid_recist(node)), key=lambda n: n.node_id)
        if len(valid) != 1:
            ambiguous_studies += 1
            if ambiguity_policy == "exclude":
                return [], ambiguous_studies
        if valid:
            selected.append(valid[0])
        elif ambiguity_policy == "deterministic":
            # Retained only for the construction audit; downstream loading
            # rejects trajectories without five valid RECIST measurements.
            selected.append(sorted(group, key=lambda n: n.node_id)[0])
    return sorted(selected, key=lambda n: n.image.study_order), ambiguous_studies


def component_rows(
    nodes: Dict[str, LesionNode],
    uf: UnionFind,
    min_len: int,
    ambiguity_policy: str = "deterministic",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    components: Dict[str, List[LesionNode]] = defaultdict(list)
    for node_id, node in nodes.items():
        components[uf.find(node_id)].append(node)

    summary_rows: List[Dict[str, Any]] = []
    timepoint_rows: List[Dict[str, Any]] = []
    length_counter: Counter = Counter()

    sorted_components = sorted(
        components.values(),
        key=lambda group: (-len({n.image.study_key for n in group}), group[0].image.study_order),
    )

    for component_index, group in enumerate(sorted_components, start=1):
        selected, ambiguous_studies = select_one_node_per_study(group, ambiguity_policy=ambiguity_policy)
        excluded_strict_ambiguity = int(ambiguity_policy == "exclude" and not selected)
        unique_studies = len(selected)
        length_counter[unique_studies] += 1
        patient_ids = sorted({n.image.patient_id for n in group})
        split_names = sorted({n.split for n in group})
        trajectory_id = f"DLT_TRAJ_{component_index:06d}"

        summary_rows.append(
            {
                "trajectory_id": trajectory_id,
                "unique_studies": unique_studies,
                "node_count": len(group),
                "patient_count": len(patient_ids),
                "patient_ids": "|".join(patient_ids),
                "splits": "|".join(split_names),
                "ambiguity_policy": ambiguity_policy,
                "ambiguous_study_count": ambiguous_studies,
                "excluded_strict_ambiguity": excluded_strict_ambiguity,
                "eligible_len3": int(unique_studies >= 3),
                "eligible_len4": int(unique_studies >= 4),
                "eligible_len5": int(unique_studies >= 5),
                "eligible_len8": int(unique_studies >= 8),
            }
        )

        for followup_index, node in enumerate(selected):
            timepoint_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "followup_index": followup_index,
                    "ambiguity_policy": ambiguity_policy,
                    "ambiguous_study_count": ambiguous_studies,
                    "unique_studies": unique_studies,
                    "node_count": len(group),
                    "patient_id": node.image.patient_id,
                    "study_id": node.image.study_id,
                    "scan_id": node.image.scan_id,
                    "study_key": node.image.study_key,
                    "image_name": node.image.image_name,
                    "slice_start": node.image.slice_start,
                    "slice_end": node.image.slice_end,
                    "split": node.split,
                    "node_id": node.node_id,
                    "center_xyz": compact_float_list(node.center, 3),
                    "box_xyzwhd": compact_float_list(node.box, 3),
                    "spacing_xyz": compact_float_list(node.spacing, 4),
                    "long_axis_mm": node.long_axis_mm,
                    "short_axis_mm": node.short_axis_mm,
                    "area_proxy_mm2": node.area_proxy_mm2,
                    "volume_proxy_mm3": node.volume_proxy_mm3,
                    "recist_slice": node.recist_slice,
                }
            )

    len5_rows = [
        row for row in timepoint_rows if int(row["unique_studies"]) >= min_len
    ]
    return summary_rows, timepoint_rows, length_counter


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def graph_component_audit(
    nodes: Dict[str, LesionNode], uf: UnionFind, edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Report graph ambiguity without using lesion size or future outcomes."""
    members: Dict[str, List[str]] = defaultdict(list)
    for node_id in nodes:
        members[uf.find(node_id)].append(node_id)
    neighbours: Dict[str, set[str]] = defaultdict(set)
    edge_multiplicity: Counter = Counter()
    for edge in edges:
        a, b = sorted([edge["source_node_id"], edge["target_node_id"]])
        neighbours[a].add(b)
        neighbours[b].add(a)
        edge_multiplicity[(a, b)] += 1
    rows = []
    ordered = sorted(
        members.values(),
        key=lambda group: (-len({nodes[n].image.study_key for n in group}), nodes[group[0]].image.study_order),
    )
    for index, group in enumerate(ordered, start=1):
        studies = [nodes[n].image.study_key for n in group]
        study_counts = Counter(studies)
        component_edges = [
            pair for pair in edge_multiplicity if pair[0] in group and pair[1] in group
        ]
        rows.append(
            {
                "trajectory_id": f"DLT_TRAJ_{index:06d}",
                "node_count": len(group),
                "unique_studies_before_selection": len(set(studies)),
                "branching_node_count_degree_gt2": sum(len(neighbours[n]) > 2 for n in group),
                "max_node_degree": max((len(neighbours[n]) for n in group), default=0),
                "studies_with_multiple_candidate_nodes": sum(v > 1 for v in study_counts.values()),
                "duplicate_pair_annotations": sum(max(edge_multiplicity[pair] - 1, 0) for pair in component_edges),
                "ambiguous_component": int(
                    any(len(neighbours[n]) > 2 for n in group) or any(v > 1 for v in study_counts.values())
                ),
            }
        )
    return rows


def write_report(
    path: Path,
    n_pairs: int,
    n_nodes: int,
    n_edges: int,
    summary_rows: List[Dict[str, Any]],
    length_counter: Counter,
    min_len: int,
) -> None:
    counts = {
        ">=3": sum(int(row["eligible_len3"]) for row in summary_rows),
        "=4": sum(1 for row in summary_rows if int(row["unique_studies"]) == 4),
        ">=4": sum(int(row["eligible_len4"]) for row in summary_rows),
        ">=5": sum(int(row["eligible_len5"]) for row in summary_rows),
        ">=8": sum(int(row["eligible_len8"]) for row in summary_rows),
    }
    exact_lengths = ", ".join(
        f"{length}:{count}" for length, count in sorted(length_counter.items())
    )
    lines = [
        "# DeepLesion/DLT Longitudinal Trajectory Validation",
        "",
        "## Inputs",
        "",
        f"- DLT matched lesion pairs loaded: {n_pairs}",
        f"- Graph nodes: {n_nodes}",
        f"- Graph edges: {n_edges}",
        f"- Connected components / candidate lesion tracks: {len(summary_rows)}",
        "",
        "## Follow-up Length Counts",
        "",
        "| Criterion | Number of candidate trajectories |",
        "|---|---:|",
        f"| length >= 3 | {counts['>=3']} |",
        f"| length = 4 | {counts['=4']} |",
        f"| length >= 4 | {counts['>=4']} |",
        f"| length >= 5 | {counts['>=5']} |",
        f"| length >= 8 | {counts['>=8']} |",
        "",
        "## Exact Length Distribution",
        "",
        exact_lengths if exact_lengths else "No components found.",
        "",
        "## Interpretation",
        "",
        (
            f"If length >= {min_len} contains dozens to hundreds of trajectories, "
            "DeepLesion/DLT can be used as the main real multi-follow-up dataset. "
            "If the count is small, keep NLSTt as the main real sparse dataset and use "
            "DeepLesion/DLT as an auxiliary validation cohort."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate same-lesion 5-follow-up trajectories from DLT annotations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deeplesion_dlt"),
        help="Directory containing DLT train/valid/test JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_deeplesion_longitudinal"),
        help="Output directory for CSV summaries.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to read. DLT uses valid.json; val.json is also accepted.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing DLT JSON files.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=5,
        help="Minimum unique-study count for the exported candidate trajectory table.",
    )
    parser.add_argument(
        "--node-precision",
        type=int,
        default=1,
        help="Decimal precision used when fingerprinting lesion geometry.",
    )
    parser.add_argument(
        "--ambiguity-policy",
        choices=["deterministic", "exclude"],
        default="deterministic",
        help=(
            "deterministic: valid RECIST then stable node-id tie-break; "
            "exclude: strict cohort excluding any component with an ambiguous study"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["valid" if split == "val" else split for split in args.splits]

    if not args.no_download:
        download_missing_json(args.data_dir, splits)

    pairs = load_pairs(args.data_dir, splits)
    if not pairs:
        raise SystemExit(
            f"No DLT annotation pairs loaded. Put train/valid/test.json in {args.data_dir} "
            "or run without --no-download."
        )

    nodes, uf, edges = build_graph(pairs, precision=args.node_precision)
    summary_rows, timepoint_rows, length_counter = component_rows(
        nodes, uf, min_len=args.min_len, ambiguity_policy=args.ambiguity_policy
    )
    len_rows = [
        row for row in timepoint_rows if int(row["unique_studies"]) >= args.min_len
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_strict" if args.ambiguity_policy == "exclude" else ""
    write_csv(args.out_dir / f"deeplesion_trajectory_summary{suffix}.csv", summary_rows)
    write_csv(args.out_dir / f"deeplesion_trajectory_timepoints{suffix}.csv", timepoint_rows)
    audit_rows = graph_component_audit(nodes, uf, edges)
    write_csv(args.out_dir / f"deeplesion_component_graph_audit{suffix}.csv", audit_rows)
    write_csv(args.out_dir / f"deeplesion_trajectories_len{args.min_len}{suffix}.csv", len_rows)
    if args.min_len == 5 and not suffix:
        write_csv(args.out_dir / "deeplesion_trajectories_len5.csv", len_rows)
    write_report(
        args.out_dir / "deeplesion_longitudinal_report.md",
        n_pairs=len(pairs),
        n_nodes=len(nodes),
        n_edges=len(edges),
        summary_rows=summary_rows,
        length_counter=length_counter,
        min_len=args.min_len,
    )

    counts = {
        ">=3": sum(int(row["eligible_len3"]) for row in summary_rows),
        "=4": sum(1 for row in summary_rows if int(row["unique_studies"]) == 4),
        ">=5": sum(int(row["eligible_len5"]) for row in summary_rows),
        ">=8": sum(int(row["eligible_len8"]) for row in summary_rows),
    }
    print("\nDeepLesion/DLT longitudinal validation")
    print(f"Pairs: {len(pairs)}")
    print(f"Nodes: {len(nodes)}")
    print(f"Candidate trajectories: {len(summary_rows)}")
    print(f"Ambiguous graph components: {sum(int(row['ambiguous_component']) for row in audit_rows)}")
    print(f"Length >= 3: {counts['>=3']}")
    print(f"Length = 4: {counts['=4']}")
    print(f"Length >= 5: {counts['>=5']}")
    print(f"Length >= 8: {counts['>=8']}")
    print(f"Wrote: {args.out_dir / 'deeplesion_longitudinal_report.md'}")


if __name__ == "__main__":
    main()
