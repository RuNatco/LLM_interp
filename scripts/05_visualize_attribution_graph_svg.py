from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any


Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])


class InterventionGraph:
    def __init__(self, ordered_nodes: list[list["Supernode"]], prompt: str):
        self.ordered_nodes = ordered_nodes
        self.prompt = prompt
        self.nodes: dict[str, Supernode] = {}


class Supernode:
    def __init__(
        self,
        name: str,
        features: list[Feature],
        children: list["Supernode"] | None = None,
        intervention: str | None = None,
        replacement_node: "Supernode | None" = None,
        activation: float | None = None,
        effect: float | None = None,
    ):
        self.name = name
        self.features = features
        self.activation = activation
        self.default_activations = None
        self.children = children or []
        self.intervention = intervention
        self.replacement_node = replacement_node
        self.effect = effect

    def __repr__(self) -> str:
        return (
            f"Supernode(name={self.name!r}, activation={self.activation}, "
            f"effect={self.effect}, children={len(self.children)})"
        )


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _short_target_name(name: str, max_len: int = 34) -> str:
    name = name.replace("logit_diff:", "Δlogit ")
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def load_v0_graph(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = {"nodes", "targets", "adjacency_matrix"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"Graph JSON does not look like qwen_clt v0 graph. Missing keys: {sorted(missing)}"
        )
    return data


def graph_json_to_intervention_graph(
    graph_data: dict[str, Any],
    prompt: str,
    target_index: int = 0,
    top_k: int = 24,
    max_features_per_layer: int = 6,
    min_abs_effect: float = 0.0,
) -> tuple[InterventionGraph, list[tuple[str, float]], list[dict[str, Any]]]:
    """Convert v0 graph JSON into a small Supernode graph.

    Layout convention:
      row 0..N: top feature nodes grouped by source layer;
      final row: selected target node.

    Each selected feature points to the target. The node activation is normalized
    by the maximum absolute activation among selected features; edge sign is
    encoded in the intervention badge: positive `+`, negative `-`.
    """
    nodes = graph_data["nodes"]
    targets = graph_data["targets"]
    adjacency = graph_data["adjacency_matrix"]

    if not targets:
        raise ValueError("Graph contains no targets.")
    if target_index < 0 or target_index >= len(targets):
        raise IndexError(f"target_index={target_index} is out of range 0..{len(targets)-1}")

    target_name = str(targets[target_index].get("name", f"target_{target_index}"))
    target_effects = adjacency[target_index]

    rows = []
    for i, node in enumerate(nodes):
        effect = _safe_float(target_effects[i] if i < len(target_effects) else 0.0)
        if abs(effect) < min_abs_effect:
            continue
        enriched = dict(node)
        enriched["effect"] = effect
        enriched["abs_effect"] = abs(effect)
        rows.append(enriched)

    rows.sort(key=lambda n: n["abs_effect"], reverse=True)
    selected = rows[:top_k]

    if max_features_per_layer > 0:
        capped = []
        counts: dict[int, int] = defaultdict(int)
        for node in selected:
            layer = int(node.get("layer", -1))
            if counts[layer] < max_features_per_layer:
                capped.append(node)
                counts[layer] += 1
        selected = capped

    max_abs_activation = max((_safe_float(n.get("activation"), 0.0) for n in selected), default=1.0)
    if max_abs_activation <= 0:
        max_abs_activation = 1.0

    target_node = Supernode(
        name=_short_target_name(target_name),
        features=[],
        activation=None,
        effect=sum(_safe_float(n.get("effect"), 0.0) for n in selected),
    )

    by_layer: dict[int, list[Supernode]] = defaultdict(list)
    for node in selected:
        layer = int(node.get("layer", -1))
        pos = int(node.get("pos", -1))
        feature_idx = int(node.get("feature_idx", -1))
        activation = _safe_float(node.get("activation"), 0.0)
        effect = _safe_float(node.get("effect"), 0.0)

        name = f"L{layer}/P{pos}/F{feature_idx}"
        sign_badge = f"{effect:+.2e}"
        supernode = Supernode(
            name=name,
            features=[Feature(layer=layer, pos=pos, feature_idx=feature_idx)],
            children=[target_node],
            activation=max(0.0, min(abs(activation) / max_abs_activation, 1.0)),
            intervention=sign_badge,
            effect=effect,
        )
        by_layer[layer].append(supernode)

    ordered_nodes: list[list[Supernode]] = []
    for layer in sorted(by_layer.keys()):
        layer_nodes = sorted(by_layer[layer], key=lambda n: abs(n.effect or 0.0), reverse=True)
        ordered_nodes.append(layer_nodes)
    ordered_nodes.append([target_node])

    top_outputs = [
        (_short_target_name(target_name), abs(sum(_safe_float(n.get("effect"), 0.0) for n in selected)))
    ]

    return InterventionGraph(ordered_nodes=ordered_nodes, prompt=prompt), top_outputs, selected


def calculate_node_positions(nodes: list[list[Supernode]]):
    """Calculate positions for all nodes including replacements.
    """
    node_width = 112
    node_height = 35
    horizontal_gap = 42
    vertical_gap = 68

    max_row_len = max((len(row) for row in nodes), default=1)
    container_width = max(700, max_row_len * node_width + (max_row_len - 1) * horizontal_gap + 120)
    container_height = max(300, len(nodes) * vertical_gap + 80)

    node_data: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(nodes):
        row_y = 70 + row_index * vertical_gap
        row_width = len(row) * node_width + max(0, len(row) - 1) * horizontal_gap
        start_x = (container_width - row_width) / 2
        for col_index, node in enumerate(row):
            node_x = start_x + col_index * (node_width + horizontal_gap)
            node_data[node.name] = {
                "x": node_x,
                "y": row_y,
                "node": node,
                "width": node_width,
                "height": node_height,
            }

    all_nodes = set()
    for row in nodes:
        for node in row:
            all_nodes.add(node)
            if node.replacement_node:
                all_nodes.add(node.replacement_node)

    for node in all_nodes:
        if node.replacement_node and node.replacement_node.name not in node_data:
            original_pos = node_data.get(node.name)
            if original_pos:
                node_data[node.replacement_node.name] = {
                    "x": original_pos["x"] + 30,
                    "y": original_pos["y"] - 35,
                    "node": node.replacement_node,
                    "width": node_width,
                    "height": node_height,
                }

    return node_data, container_width, container_height


def get_node_center(node_data, node_name):
    node = node_data.get(node_name)
    if not node:
        return {"x": 0, "y": 0}
    return {
        "x": node["x"] + node.get("width", 112) / 2,
        "y": node["y"] + node.get("height", 35) / 2,
    }


def build_connections_data(nodes: list[list[Supernode]]):
    connections = []
    all_nodes = set()

    def add_node_and_related(node):
        all_nodes.add(node)
        if node.replacement_node:
            add_node_and_related(node.replacement_node)
        for child in node.children:
            add_node_and_related(child)

    for row in nodes:
        for node in row:
            add_node_and_related(node)

    replacement_nodes = set()
    for node in all_nodes:
        if node.replacement_node:
            replacement_nodes.add(node.replacement_node.name)

    for node in all_nodes:
        for child in node.children:
            if node.replacement_node:
                continue
            connection = {"from": node.name, "to": child.name, "effect": node.effect}
            if node.name in replacement_nodes:
                connection["replacement"] = True
            connections.append(connection)
    return connections


def create_connection_svg(node_data, connections):
    svg_parts = []
    max_abs_effect = max((abs(_safe_float(c.get("effect"), 0.0)) for c in connections), default=1.0)
    if max_abs_effect <= 0:
        max_abs_effect = 1.0

    for conn in connections:
        from_center = get_node_center(node_data, conn["from"])
        to_center = get_node_center(node_data, conn["to"])
        if from_center["x"] == 0 or to_center["x"] == 0:
            continue

        effect = _safe_float(conn.get("effect"), 0.0)
        stroke_color = "#8B4513" if effect >= 0 else "#555"
        stroke_width = str(1.5 + 4.5 * min(abs(effect) / max_abs_effect, 1.0))
        opacity = str(0.35 + 0.65 * min(abs(effect) / max_abs_effect, 1.0))

        svg_parts.append(
            f'<line x1="{from_center["x"]}" y1="{from_center["y"]}" '
            f'x2="{to_center["x"]}" y2="{to_center["y"]}" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
        )

        dx = to_center["x"] - from_center["x"]
        dy = to_center["y"] - from_center["y"]
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            arrow_size = 8
            arrow_tip_x = to_center["x"]
            arrow_tip_y = to_center["y"]
            base_x = arrow_tip_x - arrow_size * dx_norm
            base_y = arrow_tip_y - arrow_size * dy_norm
            perp_x = -dy_norm * (arrow_size / 2)
            perp_y = dx_norm * (arrow_size / 2)
            left_x = base_x + perp_x
            left_y = base_y + perp_y
            right_x = base_x - perp_x
            right_y = base_y - perp_y
            svg_parts.append(
                f'<polygon points="{arrow_tip_x},{arrow_tip_y} {left_x},{left_y} {right_x},{right_y}" '
                f'fill="{stroke_color}" opacity="{opacity}"/>'
            )
    return "\n".join(svg_parts)


def create_nodes_svg(node_data):
    svg_parts = []
    replacement_nodes = set()
    for data in node_data.values():
        node = data["node"]
        if node.replacement_node:
            replacement_nodes.add(node.replacement_node.name)

    for name, data in node_data.items():
        node = data["node"]
        x = data["x"]
        y = data["y"]
        w = data.get("width", 112)
        h = data.get("height", 35)

        is_low_activation = node.activation is not None and node.activation <= 0.25
        is_replacement = name in replacement_nodes
        is_target = not node.features
        has_negative_effect = node.effect is not None and node.effect < 0

        if is_target:
            fill_color = "#FFF8DC"
            text_color = "#333"
            stroke_color = "#D2691E"
        elif is_low_activation:
            fill_color = "#f0f0f0"
            text_color = "#888"
            stroke_color = "#ddd"
        elif is_replacement:
            fill_color = "#FFF8DC"
            text_color = "#333"
            stroke_color = "#D2691E"
        elif has_negative_effect:
            fill_color = "#eeeeee"
            text_color = "#333"
            stroke_color = "#777"
        else:
            fill_color = "#e8e8e8"
            text_color = "#333"
            stroke_color = "#999"

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" rx="8"/>'
        )

        escaped_name = html.escape(name)
        svg_parts.append(
            f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" '
            f'fill="{text_color}" font-family="Arial, sans-serif" font-size="11" font-weight="bold">{escaped_name}</text>'
        )

        if node.activation is not None:
            activation_pct = round(node.activation * 100)
            label_x = x - 12
            label_y = y - 8
            svg_parts.append(
                f'<rect x="{label_x}" y="{label_y}" width="34" height="16" '
                f'fill="white" stroke="#ccc" stroke-width="1" rx="4"/>'
            )
            svg_parts.append(
                f'<text x="{label_x + 17}" y="{label_y + 12}" text-anchor="middle" '
                f'fill="#8B4513" font-family="Arial, sans-serif" font-size="10" font-weight="bold">{activation_pct}%</text>'
            )

        if node.intervention:
            intervention_x = x + w - 58
            intervention_y = y - 9
            text_width = max(58, min(len(node.intervention) * 7 + 10, 86))
            escaped_intervention = html.escape(node.intervention)
            badge_color = "#8B4513" if not node.intervention.startswith("-") else "#666"
            svg_parts.append(
                f'<rect x="{intervention_x}" y="{intervention_y}" width="{text_width}" height="16" '
                f'fill="{badge_color}" stroke="none" rx="12"/>'
            )
            svg_parts.append(
                f'<text x="{intervention_x + text_width / 2}" y="{intervention_y + 12}" text-anchor="middle" '
                f'fill="white" font-family="Arial, sans-serif" font-size="9" font-weight="bold">{escaped_intervention}</text>'
            )
    return "\n".join(svg_parts)


def wrap_text_for_svg(text, max_width=95):
    if len(text) <= max_width:
        return [text]
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_width:
            current_line = current_line + " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def create_graph_visualization_svg(
    intervention_graph: InterventionGraph,
    top_outputs: list[tuple[str, float]],
    selected_features: list[dict[str, Any]],
) -> str:
    nodes = intervention_graph.ordered_nodes
    prompt = intervention_graph.prompt
    node_data, width, graph_height = calculate_node_positions(nodes)
    connections = build_connections_data(nodes)
    connections_svg = create_connection_svg(node_data, connections)
    nodes_svg = create_nodes_svg(node_data)

    prompt_y = graph_height + 35
    output_y_start = prompt_y + 65
    total_height = output_y_start + 75

    # Output summary badges.
    output_items_svg = []
    current_x = 40
    for i, (text, score) in enumerate(top_outputs[:6]):
        display_text = text if text else "(empty)"
        escaped_display_text = html.escape(display_text)
        score_text = f"|effect|={score:.3e}"
        item_width = max(160, min(len(display_text) * 8 + len(score_text) * 6 + 24, 360))
        output_items_svg.append(
            f'<rect x="{current_x}" y="{output_y_start}" width="{item_width}" height="22" '
            f'fill="#e8e8e8" stroke="none" rx="6"/>'
        )
        output_items_svg.append(
            f'<text x="{current_x + 8}" y="{output_y_start + 15}" '
            f'fill="#333" font-family="Arial, sans-serif" font-size="11" font-weight="bold">'
            f'{escaped_display_text} <tspan fill="#555" font-size="10">{score_text}</tspan></text>'
        )
        current_x += item_width + 10
    output_items_svg_str = "\n".join(output_items_svg)

    escaped_prompt = html.escape(prompt)
    prompt_lines = wrap_text_for_svg(escaped_prompt, max_width=max(80, int(width / 8)))
    prompt_text_svg = []
    for i, line in enumerate(prompt_lines[:3]):
        y_offset = prompt_y + 35 + (i * 15)
        prompt_text_svg.append(
            f'<text x="40" y="{y_offset}" fill="#333" font-family="Arial, sans-serif" font-size="12">{line}</text>'
        )
    prompt_text_svg_str = "\n".join(prompt_text_svg)

    n_features = len(selected_features)
    feature_summary = f"Selected top features: {n_features}"

    svg_content = f"""<svg width="{width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="{width}" height="{total_height}" fill="#f5f5f5"/>
    <rect x="20" y="20" width="{width - 40}" height="{total_height - 40}" fill="white" stroke="none" rx="12"/>

    <text x="40" y="45" fill="#666" font-family="Arial, sans-serif" font-size="14" font-weight="bold"
          text-transform="uppercase" letter-spacing="1px">Qwen CLT Attribution Graph</text>
    <text x="40" y="63" fill="#888" font-family="Arial, sans-serif" font-size="11">{html.escape(feature_summary)}</text>

    <g>
        {connections_svg}
        {nodes_svg}
    </g>

    <line x1="40" y1="{prompt_y}" x2="{width - 40}" y2="{prompt_y}" stroke="#ddd" stroke-width="1"/>
    <text x="40" y="{prompt_y + 20}" fill="#666" font-family="Arial, sans-serif" font-size="12" font-weight="bold"
          text-transform="uppercase" letter-spacing="0.5px">Prompt</text>
    {prompt_text_svg_str}

    <text x="40" y="{output_y_start - 10}" fill="#666" font-family="Arial, sans-serif" font-size="10" font-weight="bold"
          text-transform="uppercase" letter-spacing="0.5px">Target / effect summary</text>
    {output_items_svg_str}
</svg>"""
    return svg_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Render qwen_clt attribution graph JSON to SVG.")
    parser.add_argument("--graph", required=True, help="Path to graph JSON from scripts/03_build_attribution_graph.py")
    parser.add_argument("--output", default=None, help="Output SVG path. Default: <graph_stem>_viz.svg")
    parser.add_argument("--prompt", default="", help="Prompt text to show under the graph")
    parser.add_argument("--target-index", type=int, default=0, help="Target row index in adjacency_matrix")
    parser.add_argument("--top-k", type=int, default=24, help="Number of highest-|effect| feature nodes to show")
    parser.add_argument(
        "--max-features-per-layer",
        type=int,
        default=6,
        help="Cap selected features per layer. Use 0 to disable cap.",
    )
    parser.add_argument("--min-abs-effect", type=float, default=0.0, help="Drop features below this |effect|")
    parser.add_argument(
        "--save-selected-json",
        default=None,
        help="Optional path to save selected top feature rows as JSON",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph)
    graph_data = load_v0_graph(graph_path)
    prompt = args.prompt or graph_data.get("prompt", "")

    intervention_graph, top_outputs, selected = graph_json_to_intervention_graph(
        graph_data=graph_data,
        prompt=prompt,
        target_index=args.target_index,
        top_k=args.top_k,
        max_features_per_layer=args.max_features_per_layer,
        min_abs_effect=args.min_abs_effect,
    )

    svg = create_graph_visualization_svg(intervention_graph, top_outputs, selected)

    output = Path(args.output) if args.output else graph_path.with_name(graph_path.stem + "_viz.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")

    if args.save_selected_json:
        selected_path = Path(args.save_selected_json)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved SVG visualization to: {output}")
    print(f"Selected features: {len(selected)}")


if __name__ == "__main__":
    main()
