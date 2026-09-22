"""Dataset for CoVLM-Bench single-stage VLM baseline."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .evidence import FEATURE_DIM as L1_EVIDENCE_FEATURE_DIM
from .evidence import ROUTE_FEATURE_DIM as ROUTE_EVIDENCE_FEATURE_DIM
from .evidence import (
    build_l1_evidence,
    build_route_conditioned_l1_evidence,
    load_l1_records,
    select_route_conditioned_evidence_records,
)
from .geometry import build_geov2x_geometry


COMMANDS = ["GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "LATERAL_SHIFT", "STOP", "SLOW_DOWN", "UNKNOWN"]
COMMAND2ID = {name: idx for idx, name in enumerate(COMMANDS)}
ID2COMMAND = {idx: name for name, idx in COMMAND2ID.items()}
OBJECT_REF_RE = re.compile(r"\b[EI]\d+\b")
LEGACY_COMMAND_ALIASES = {"MERGE": "LATERAL_SHIFT"}
DEFAULT_PART1_OBJECT_LIMIT = 6
DEFAULT_PART1_SENTENCE_LIMIT = 3


def canonical_command(command: str) -> str:
    text = str(command or "UNKNOWN").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "STRAIGHT": "GO_STRAIGHT",
        "FORWARD": "GO_STRAIGHT",
        "KEEP_STRAIGHT": "GO_STRAIGHT",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
        "MERGE": "LATERAL_SHIFT",
        "LANE_CHANGE": "LATERAL_SHIFT",
        "CHANGE_LANE": "LATERAL_SHIFT",
        "BRAKE": "STOP",
        "DECELERATE": "SLOW_DOWN",
    }
    return aliases.get(text, text if text in COMMAND2ID else "UNKNOWN")


def default_prompt(
    mode: str,
    ego_status: Optional[Dict[str, Any]] = None,
    part1_object_limit: int = 0,
    part1_sentence_limit: int = 0,
    neutral_ego_only_prompt: bool = False,
    view_order: str = "ego_infra",
) -> str:
    order = str(view_order or "ego_infra").lower()
    if order not in {"ego_infra", "infra_ego"}:
        raise ValueError("view_order must be ego_infra or infra_ego")
    if mode == "ego_only" and neutral_ego_only_prompt:
        base = [
            "You are an autonomous driving assistant for CoVLM-Bench.",
            "Only the ego-vehicle front camera image is available.",
            "Reason only from objects and road context visible from the ego front camera view.",
        ]
    else:
        base = [
            "You are a V2X-aware autonomous driving assistant for CoVLM-Bench.",
            (
                "Image 2 is the ego-vehicle front camera view."
                if mode == "v2x_image" and order == "infra_ego"
                else "Image 1 is the ego-vehicle front camera view."
            ),
        ]
    if mode == "ego_only":
        if not neutral_ego_only_prompt:
            base.append("Only Image 1 is provided; reason from the ego camera view only.")
    elif mode == "v2x_feature":
        base.extend(
            [
                "Image 2 is represented by cached/compressed infrastructure-side visual tokens when available.",
                "Use the infrastructure-side feature context to reason about V2X-only or occluded objects.",
            ]
        )
    else:
        base.extend(
            [
                (
                    "Image 1 is the infrastructure-side camera view."
                    if order == "infra_ego"
                    else "Image 2 is the infrastructure-side camera view."
                ),
                "Use the infrastructure view only when it provides reliable additional context; otherwise rely on the ego view.",
            ]
        )
    part2_label = (
        "critical objects visible from the ego view"
        if mode == "ego_only" and neutral_ego_only_prompt
        else "V2X-aware critical objects"
    )
    base.extend(
        [
            "Output the response in the required CoVLM-Bench reasoning format.",
            (
                f"Required output: Part 1 scene overview, Part 2 {part2_label}, "
                "and Part 3 decision reasoning."
            ),
            (
                "Do not output a separate action section, command field, or numeric "
                "waypoints in text; command and trajectory are predicted by independent "
                "structured heads."
            ),
        ]
    )
    if mode == "ego_only" and neutral_ego_only_prompt:
        base.append("Do not infer infrastructure-only evidence or refer to any unavailable infrastructure image.")
    constraints: List[str] = []
    if part1_sentence_limit > 0:
        constraints.append(f"at most {part1_sentence_limit} sentences")
    if part1_object_limit > 0:
        constraints.append(f"at most {part1_object_limit} object mentions")
    if constraints:
        base.append(
            "Compactness constraint: Part 1 must be "
            + " and ".join(constraints)
            + "; summarize remaining traffic collectively and move on instead of enumerating every visible actor."
        )
        base.append("Always include all three reasoning parts within the generation budget.")
    if not constraints:
        base.append(
            "Compactness constraint: Part 1 should stay short, summarize the overall scene first, "
            "and avoid long object listings unless they are essential for the decision."
        )
    if ego_status:
        available = bool(ego_status.get("available", False))
        if available:
            speed = float(ego_status.get("speed_mps", 0.0))
            yaw_rate = ego_status.get("yaw_rate_radps")
            yaw_text = "unknown" if yaw_rate is None else f"{float(yaw_rate):.3f} rad/s"
            base.append(
                "Ego status from current and previous pose only: "
                f"available=yes; current speed={speed:.2f} m/s; yaw rate={yaw_text}."
            )
        else:
            base.append(
                "Ego status from current and previous pose only: available=no; use visual context without ego-speed conditioning."
            )
    return "\n".join(base)


def split_part1(text: str) -> Optional[Tuple[str, str, str]]:
    marker = "Part 1 - Scene overview:"
    start = text.find(marker)
    if start < 0:
        return None
    next_match = re.search(r"\n\s*Part 2\b", text[start:])
    if not next_match:
        return None
    part1_end = start + next_match.start()
    return text[:start], text[start:part1_end].strip(), text[part1_end:]


def count_object_refs(text: str) -> int:
    return len(OBJECT_REF_RE.findall(text))


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def compact_part1_block(
    part1: str,
    object_limit: int,
    sentence_limit: int,
) -> str:
    if object_limit <= 0 and sentence_limit <= 0:
        return part1
    if "\n" not in part1:
        return part1
    header, body = part1.split("\n", 1)
    sentences = split_sentences(body)
    if not sentences:
        return part1

    kept: List[str] = []
    object_count = 0
    omitted_objects = 0
    for sentence in sentences:
        refs = count_object_refs(sentence)
        would_exceed_objects = object_limit > 0 and refs > 0 and object_count + refs > object_limit
        would_exceed_sentences = sentence_limit > 0 and len(kept) >= sentence_limit
        if would_exceed_objects or would_exceed_sentences:
            omitted_objects += refs
            continue
        kept.append(sentence)
        object_count += refs

    if not kept:
        kept = sentences[:1]
    if omitted_objects > 0 or len(kept) < len(sentences):
        summary = (
            "Additional visible traffic participants are present and should be treated as grouped "
            "background context unless Part 2 marks them as critical."
        )
        if sentence_limit <= 0 or len(kept) < sentence_limit:
            kept.append(summary)
        else:
            kept[-1] = summary
    return header + "\n" + " ".join(kept)


def compact_l4_target_text(text: str, object_limit: int, sentence_limit: int) -> str:
    split = split_part1(text)
    if split is None:
        return text
    prefix, part1, suffix = split
    compact_part1 = compact_part1_block(part1, object_limit, sentence_limit)
    return prefix + compact_part1 + suffix


class CoVLABaselineDataset:
    """JSONL-backed CoVLM L4 dataset.

    Args:
        index_path: JSONL generated by ``data.build_index``.
        mode: ``ego_only``, ``v2x_image`` or ``v2x_feature``.
        split: Optional split filter.
        load_images: If true, ``__getitem__`` returns PIL images; otherwise paths only.
        no_cot: If true, train only on action/command/waypoint text.
    """

    def __init__(
        self,
        index_path: str,
        mode: str = "v2x_image",
        split: Optional[str] = None,
        load_images: bool = False,
        no_cot: bool = False,
        use_ego_status_prompt: bool = False,
        part1_object_limit: int = 0,
        part1_sentence_limit: int = 0,
        compact_target_part1: bool = False,
        base_prediction_path: Optional[str] = None,
        use_geov2x_geometry: bool = False,
        geov2x_geometry_zero: bool = False,
        v2x_root: str = "data/V2X-Seq-SPD",
        infra_condition: str = "normal",
        infra_condition_mix: Optional[Sequence[str]] = None,
        shuffled_infra_map_path: Optional[str] = None,
        use_l1_evidence_tokens: bool = False,
        evidence_top_k: int = 12,
        evidence_mode: str = "l1",
        route_corridor_width_m: float = 6.0,
        route_near_horizon_steps: int = 6,
        route_relevance_variant: str = "strict",
        teacher_prediction_path: Optional[str] = None,
        route_risk_supervision: bool = False,
        neutral_ego_only_prompt: bool = False,
        use_trajectory_grounded_role: bool = False,
        trajectory_relevant_repeat: int = 1,
        view_order: str = "ego_infra",
        teacher_exclude_decision_relevant: bool = False,
        causal_ego_anchor: bool = False,
    ) -> None:
        if mode not in {"ego_only", "v2x_image", "v2x_feature"}:
            raise ValueError(f"Unsupported mode={mode}; expected ego_only, v2x_image, or v2x_feature")
        self.index_path = Path(index_path)
        self.mode = mode
        self.split = split
        self.load_images = load_images
        self.no_cot = no_cot
        self.use_ego_status_prompt = use_ego_status_prompt
        self.neutral_ego_only_prompt = bool(neutral_ego_only_prompt)
        self.part1_object_limit = max(
            0,
            int(DEFAULT_PART1_OBJECT_LIMIT if part1_object_limit is None else part1_object_limit),
        )
        self.part1_sentence_limit = max(
            0,
            int(DEFAULT_PART1_SENTENCE_LIMIT if part1_sentence_limit is None else part1_sentence_limit),
        )
        self.compact_target_part1 = compact_target_part1
        self.use_geov2x_geometry = bool(use_geov2x_geometry)
        self.geov2x_geometry_zero = bool(geov2x_geometry_zero)
        self.v2x_root = str(v2x_root)
        self.infra_condition = str(infra_condition or "normal").lower()
        if self.infra_condition not in {"normal", "blank", "shuffled"}:
            raise ValueError("infra_condition must be normal, blank, or shuffled")
        self.infra_condition_mix = [str(x).lower() for x in (infra_condition_mix or [])]
        bad_mix = [x for x in self.infra_condition_mix if x not in {"normal", "blank", "shuffled"}]
        if bad_mix:
            raise ValueError(f"infra_condition_mix contains unsupported values: {bad_mix}")
        self.view_order = str(view_order or "ego_infra").lower()
        if self.view_order not in {"ego_infra", "infra_ego"}:
            raise ValueError("view_order must be ego_infra or infra_ego")
        self.causal_ego_anchor = bool(causal_ego_anchor)
        if self.causal_ego_anchor and (
            self.mode != "v2x_image" or self.view_order != "ego_infra"
        ):
            raise ValueError(
                "causal_ego_anchor requires mode=v2x_image and view_order=ego_infra"
            )
        self.shuffled_infra_map: Dict[str, str] = {}
        if shuffled_infra_map_path:
            map_path = Path(shuffled_infra_map_path)
            if not map_path.exists():
                raise FileNotFoundError(f"shuffled_infra_map_path does not exist: {map_path}")
            with map_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get("infra_image") or row.get("shuffled_infra_image")
                    if value:
                        for key_name in ("sample_id", "token"):
                            key = row.get(key_name)
                            if key is not None:
                                self.shuffled_infra_map[str(key)] = str(value)
        self.use_l1_evidence_tokens = bool(use_l1_evidence_tokens)
        self.evidence_top_k = max(1, int(evidence_top_k))
        self.evidence_mode = str(evidence_mode or "l1").lower()
        if self.evidence_mode not in {"l1", "route_conditioned"}:
            raise ValueError(f"Unsupported evidence_mode={evidence_mode!r}")
        self.route_corridor_width_m = float(route_corridor_width_m)
        self.route_near_horizon_steps = int(route_near_horizon_steps)
        self.route_relevance_variant = str(route_relevance_variant or "strict").lower()
        self.route_risk_supervision = bool(route_risk_supervision)
        self.use_trajectory_grounded_role = bool(use_trajectory_grounded_role)
        self.teacher_exclude_decision_relevant = bool(
            teacher_exclude_decision_relevant
        )
        if self.teacher_exclude_decision_relevant and not self.use_trajectory_grounded_role:
            raise ValueError(
                "teacher_exclude_decision_relevant requires use_trajectory_grounded_role=true"
            )
        self.trajectory_relevant_repeat = max(1, int(trajectory_relevant_repeat))
        self.base_predictions: Dict[str, Dict[str, Any]] = {}
        if base_prediction_path:
            base_path = Path(base_prediction_path)
            if not base_path.exists():
                raise FileNotFoundError(f"base_prediction_path does not exist: {base_path}")
            with base_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    pred = json.loads(line)
                    for key_name in ("sample_id", "token"):
                        key = pred.get(key_name)
                        if key is not None:
                            self.base_predictions[str(key)] = pred
        self.teacher_predictions: Dict[str, Dict[str, Any]] = {}
        if teacher_prediction_path:
            teacher_path = Path(teacher_prediction_path)
            if not teacher_path.exists():
                raise FileNotFoundError(f"teacher_prediction_path does not exist: {teacher_path}")
            with teacher_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    pred = json.loads(line)
                    for key_name in ("sample_id", "token"):
                        key = pred.get(key_name)
                        if key is not None:
                            self.teacher_predictions[str(key)] = pred

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.index_path}. Build it with: "
                "python -m projects.covla_baseline.data.build_index --l4-dir <L4_DIR> --output <INDEX_JSONL>"
            )
        self.items: List[Dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if split is not None and item.get("split") != split:
                    continue
                if self.use_trajectory_grounded_role:
                    role = item.get("trajectory_grounded_infra_role")
                    if role is None:
                        l4_path = Path(str(item.get("l4_path", "")))
                        if not l4_path.is_file():
                            raise FileNotFoundError(
                                "trajectory-grounded role requested but l4_path is missing for "
                                f"{item.get('sample_id')}: {l4_path}"
                            )
                        role = json.loads(l4_path.read_text(encoding="utf-8")).get(
                            "trajectory_grounded_infra_role"
                        )
                    role = str(role or "").lower()
                    if role not in {
                        "context_only",
                        "no_additional_evidence",
                        "decision_relevant",
                    }:
                        raise ValueError(
                            f"Invalid trajectory_grounded_infra_role={role!r} for "
                            f"{item.get('sample_id')}"
                        )
                    item["trajectory_grounded_infra_role"] = role
                self.items.append(item)
        if self.infra_condition_mix:
            mixed_items: List[Dict[str, Any]] = []
            for item in self.items:
                conditions = self.infra_condition_mix
                if (
                    self.use_trajectory_grounded_role
                    and item.get("trajectory_grounded_infra_role") == "decision_relevant"
                ):
                    conditions = ["normal"] * self.trajectory_relevant_repeat
                for condition in conditions:
                    clone = dict(item)
                    clone["_infra_condition_override"] = condition
                    mixed_items.append(clone)
            self.items = mixed_items
        if not self.items:
            raise RuntimeError(f"No samples loaded from {self.index_path} with split={split!r}")
        if base_prediction_path:
            missing = [
                str(item.get("sample_id") or item.get("token"))
                for item in self.items
                if self._lookup_base_prediction(item) is None
            ]
            if missing:
                raise RuntimeError(
                    f"Missing base predictions for {len(missing)} samples from {base_prediction_path}; "
                    f"examples={missing[:5]}"
                )
        if teacher_prediction_path:
            missing_teacher = [
                str(item.get("sample_id") or item.get("token"))
                for item in self.items
                if self._lookup_teacher_prediction(item) is None
            ]
            if missing_teacher:
                raise RuntimeError(
                    f"Missing teacher predictions for {len(missing_teacher)} samples from "
                    f"{teacher_prediction_path}; examples={missing_teacher[:5]}"
                )

    def __len__(self) -> int:
        return len(self.items)

    def _open_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _blank_image_like(self, path: str) -> Image.Image:
        image = self._open_image(path)
        return Image.new("RGB", image.size, (128, 128, 128))

    def _target_text(self, item: Dict[str, Any]) -> str:
        text = str(item.get("target_text", ""))
        if not self.no_cot:
            if self.compact_target_part1 or self.part1_object_limit > 0 or self.part1_sentence_limit > 0:
                return compact_l4_target_text(
                    text,
                    object_limit=self.part1_object_limit,
                    sentence_limit=self.part1_sentence_limit,
                )
            return text
        return ""

    def _lookup_base_prediction(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key_name in ("sample_id", "token"):
            key = item.get(key_name)
            if key is not None:
                pred = self.base_predictions.get(str(key))
                if pred is not None:
                    return pred
        return None

    def _lookup_teacher_prediction(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key_name in ("sample_id", "token"):
            key = item.get(key_name)
            if key is not None:
                pred = self.teacher_predictions.get(str(key))
                if pred is not None:
                    return pred
        return None

    @staticmethod
    def _validate_base_waypoints(value: Any, sample_id: Any) -> List[List[float]]:
        if not isinstance(value, list) or len(value) != 6:
            raise ValueError(f"base_waypoints for {sample_id} must have 6 steps")
        out: List[List[float]] = []
        for step in value:
            if not isinstance(step, (list, tuple)) or len(step) < 2:
                raise ValueError(f"base_waypoints for {sample_id} contains invalid step {step!r}")
            out.append([float(step[0]), float(step[1])])
        return out

    @staticmethod
    def _validate_base_logits(value: Any, sample_id: Any) -> List[float]:
        if not isinstance(value, list) or len(value) != len(COMMANDS):
            raise ValueError(f"base_command_logits for {sample_id} must have {len(COMMANDS)} values")
        return [float(v) for v in value]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.items[idx])
        command = canonical_command(item.get("command", "UNKNOWN"))
        images: List[Image.Image] = []
        image_paths = [item["ego_image"]]
        condition = str(item.get("_infra_condition_override", self.infra_condition)).lower()
        if self.mode == "v2x_image":
            infra_path = str(item["infra_image"])
            if condition == "shuffled":
                for key_name in ("sample_id", "token"):
                    key = item.get(key_name)
                    if key is not None and str(key) in self.shuffled_infra_map:
                        infra_path = self.shuffled_infra_map[str(key)]
                        break
            image_paths.append(infra_path)
        elif self.mode == "v2x_feature":
            item.setdefault("infra_feature_path", None)

        if self.load_images:
            images = [self._open_image(image_paths[0])]
            if self.mode == "v2x_image":
                if condition == "blank":
                    images.append(self._blank_image_like(str(item["infra_image"])))
                else:
                    images.append(self._open_image(image_paths[1]))

        item.update(
            {
                "mode": self.mode,
                "image_paths": image_paths,
                "infra_condition": condition,
                "view_order": self.view_order,
                "images": images,
                "prompt": default_prompt(
                    self.mode,
                    ego_status=item.get("ego_status") if self.use_ego_status_prompt else None,
                    part1_object_limit=self.part1_object_limit,
                    part1_sentence_limit=self.part1_sentence_limit,
                    neutral_ego_only_prompt=self.neutral_ego_only_prompt,
                    view_order=self.view_order,
                ),
                "part1_object_limit": self.part1_object_limit,
                "part1_sentence_limit": self.part1_sentence_limit,
                "compact_target_part1": self.compact_target_part1,
                "target_text": self._target_text(item),
                "command": command,
                "command_id": COMMAND2ID.get(command, COMMAND2ID["UNKNOWN"]),
            }
        )
        if self.causal_ego_anchor:
            item["ego_anchor_prompt"] = default_prompt(
                "ego_only",
                ego_status=item.get("ego_status") if self.use_ego_status_prompt else None,
                part1_object_limit=self.part1_object_limit,
                part1_sentence_limit=self.part1_sentence_limit,
                neutral_ego_only_prompt=True,
            )
        base_pred = self._lookup_base_prediction(item)
        if base_pred is not None:
            item["base_waypoints"] = self._validate_base_waypoints(
                base_pred.get("base_waypoints", base_pred.get("pred_waypoints")),
                item.get("sample_id"),
            )
            item["base_command_logits"] = self._validate_base_logits(
                base_pred.get("base_command_logits"),
                item.get("sample_id"),
            )
            item["base_command"] = canonical_command(
                base_pred.get("base_command", base_pred.get("pred_command", "UNKNOWN"))
            )
        if self.use_geov2x_geometry:
            geometry = build_geov2x_geometry(str(item.get("token")), v2x_root=self.v2x_root)
            item["geov2x_geometry"] = [0.0 for _ in geometry["features"]] if self.geov2x_geometry_zero else geometry["features"]
            item["geov2x_geometry_available"] = bool(geometry.get("available", False))
            item["geov2x_geometry_meta"] = {
                k: v for k, v in geometry.items() if k not in {"features"}
            }
        teacher_pred = self._lookup_teacher_prediction(item)
        if teacher_pred is not None:
            item["teacher_waypoints"] = self._validate_base_waypoints(
                teacher_pred.get("teacher_waypoints", teacher_pred.get("pred_waypoints")),
                item.get("sample_id"),
            )
            item["teacher_command_logits"] = self._validate_base_logits(
                teacher_pred.get("teacher_command_logits", teacher_pred.get("command_logits")),
                item.get("sample_id"),
            )
            item["teacher_available"] = not (
                self.teacher_exclude_decision_relevant
                and item.get("trajectory_grounded_infra_role") == "decision_relevant"
            )
        if self.route_risk_supervision:
            if "base_waypoints" not in item:
                raise ValueError(
                    f"route_risk_supervision requires base_waypoints for {item.get('sample_id')}"
                )
            records, meta = load_l1_records(str(item.get("split")), str(item.get("token")))
            _selected, summary = select_route_conditioned_evidence_records(
                records,
                item["base_waypoints"],
                top_k=self.evidence_top_k,
                corridor_width_m=self.route_corridor_width_m,
                near_horizon_steps=self.route_near_horizon_steps,
                relevance_variant=self.route_relevance_variant,
            )
            nearest = summary.get("nearest_infra_route_corridor_distance")
            nearest_norm = 0.0 if nearest is None else max(0.0, min(1.0, 1.0 - float(nearest) / 80.0))
            geometry = summary.get("route_geometry", {}) if isinstance(summary.get("route_geometry"), dict) else {}
            item["route_risk_target"] = [
                float(summary.get("route_relevance_prior", 0.0)),
                min(1.0, float(summary.get("infra_only_route_relevant_count", 0)) / 6.0),
                min(1.0, float(summary.get("shared_route_relevant_count", 0)) / 6.0),
                nearest_norm,
                float(bool(summary.get("lateral_shape_protection", False))),
                max(0.0, min(1.0, abs(float(geometry.get("lateral_delta", 0.0))) / 4.0)),
                max(0.0, min(1.0, float(geometry.get("lateral_span", 0.0)) / 6.0)),
                max(0.0, min(1.0, float(geometry.get("forward_span", 0.0)) / 60.0)),
                float(bool(geometry.get("turn_like", False))),
                float(bool(geometry.get("lateral_maneuver_like", False))),
            ]
            item["route_risk_supervision_meta"] = {
                "stage1_target_source": "current_frame_l1_route_risk_supervision_only",
                "planner_input": False,
                "num_l1_objects": int(meta.get("num_l1_objects", 0)),
            }
        if self.use_l1_evidence_tokens:
            if self.evidence_mode == "route_conditioned":
                if "base_waypoints" not in item:
                    raise ValueError(
                        f"route_conditioned evidence requires base_waypoints for {item.get('sample_id')}"
                    )
                evidence = build_route_conditioned_l1_evidence(
                    str(item.get("split")),
                    str(item.get("token")),
                    item["base_waypoints"],
                    top_k=self.evidence_top_k,
                    corridor_width_m=self.route_corridor_width_m,
                    near_horizon_steps=self.route_near_horizon_steps,
                    relevance_variant=self.route_relevance_variant,
                )
                expected_dim = ROUTE_EVIDENCE_FEATURE_DIM
            else:
                evidence = build_l1_evidence(
                    str(item.get("split")),
                    str(item.get("token")),
                    top_k=self.evidence_top_k,
                )
                expected_dim = L1_EVIDENCE_FEATURE_DIM
            if int(evidence["feature_dim"]) != expected_dim:
                raise RuntimeError(
                    f"Unexpected L1 evidence feature dim for {item.get('sample_id')}: "
                    f"{evidence['feature_dim']}"
                )
            item["object_evidence_tokens"] = evidence["tokens"]
            item["object_evidence_mask"] = evidence["mask"]
            item["v2x_relevance_prior"] = float(evidence["v2x_relevance_prior"])
            item["route_relevance_prior"] = float(evidence.get("route_relevance_prior", evidence["v2x_relevance_prior"]))
            item["object_evidence_summary"] = evidence["summary"]
            item["object_evidence_provenance"] = evidence["provenance"]
            item["lateral_shape_protection"] = bool(
                evidence["summary"].get("lateral_shape_protection", False)
            )
            item["lateral_shape_residual_scale"] = float(
                evidence["summary"].get("lateral_shape_residual_scale", 1.0)
            )
            route_geometry = evidence["summary"].get("route_geometry", {})
            item["route_right_turn_shape"] = bool(
                isinstance(route_geometry, dict)
                and route_geometry.get("turn_like")
                and route_geometry.get("lateral_direction") == "right"
            )
        return item
