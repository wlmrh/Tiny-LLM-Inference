#!/usr/bin/env python3
"""Generate the fixed-layout Tiny-LLM-Inference architecture diagram."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path


WIDTH = 1520
HEIGHT = 760
DEFAULT_OUTPUT = Path(__file__).with_name("architecture.svg")


@dataclass(frozen=True)
class Node:
    node_id: str
    x: int
    y: int
    width: int
    height: int
    title: str
    subtitle: str
    css_class: str
    shape: str = "rectangle"


NODES = (
    Node("llm", 685, 30, 150, 70, "LLM", "Offline C++ API", "entry-node"),
    Node(
        "input-preprocessor",
        65,
        370,
        250,
        72,
        "InputPreprocessor",
        "prompt → EngineCoreRequest",
        "frontend-node",
    ),
    Node(
        "scheduler",
        435,
        285,
        270,
        68,
        "Scheduler",
        "queues · prefill/decode · preemption",
        "runtime-node",
    ),
    Node(
        "kv-cache-manager",
        435,
        385,
        270,
        68,
        "KVCacheManager",
        "allocate · release",
        "runtime-node",
    ),
    Node(
        "paged-kv-cache-owner-view",
        435,
        500,
        270,
        90,
        "Paged KV cache",
        "block tables · K/V blocks",
        "storage-node",
        "cylinder",
    ),
    Node(
        "model-runner",
        815,
        285,
        270,
        68,
        "ModelRunner",
        "SchedulerOutput → ModelRunnerOutput",
        "execution-node",
    ),
    Node(
        "model",
        815,
        385,
        270,
        68,
        "LlamaForCausalLM",
        "decoder layers · PagedAttention",
        "execution-node",
    ),
    Node(
        "sampler",
        955,
        485,
        145,
        90,
        "Sampler",
        "greedy · top-k/p",
        "execution-node",
    ),
    Node(
        "paged-kv-cache-attention-view",
        800,
        485,
        145,
        90,
        "Paged KV cache",
        "attention K/V",
        "storage-node",
        "cylinder",
    ),
    Node(
        "output-preprocessor",
        1205,
        370,
        250,
        72,
        "OutPreprocessor",
        "EngineCoreOutput → UserOutput",
        "frontend-node",
    ),
)


def svg_text(x: float, y: float, value: str, css_class: str, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" class="{css_class}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def indent(value: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return prefix + value.replace("\n", "\n" + prefix)


def text_block(node: Node) -> str:
    center_x = node.x + node.width / 2
    center_y = node.y + node.height / 2
    title_y = center_y - 6
    subtitle_y = center_y + 17
    return "\n".join(
        (
            svg_text(center_x, title_y, node.title, "node-title"),
            svg_text(center_x, subtitle_y, node.subtitle, "node-subtitle"),
        )
    )


def rectangle(node: Node) -> str:
    return "\n".join(
        (
            f'<g id="{node.node_id}">',
            (
                f'  <rect x="{node.x}" y="{node.y}" width="{node.width}" height="{node.height}" '
                f'rx="10" class="{node.css_class}"/>'
            ),
            indent(text_block(node)),
            "</g>",
        )
    )


def cylinder(node: Node) -> str:
    radius_y = 11
    x2 = node.x + node.width
    bottom_y = node.y + node.height - radius_y
    path = (
        f"M {node.x} {node.y + radius_y} "
        f"C {node.x} {node.y - 4} {x2} {node.y - 4} {x2} {node.y + radius_y} "
        f"L {x2} {bottom_y} "
        f"C {x2} {bottom_y + 15} {node.x} {bottom_y + 15} {node.x} {bottom_y} Z"
    )
    return "\n".join(
        (
            f'<g id="{node.node_id}">',
            f'  <path d="{path}" class="{node.css_class}"/>',
            (
                f'  <ellipse cx="{node.x + node.width / 2:g}" cy="{node.y + radius_y}" '
                f'rx="{node.width / 2:g}" ry="{radius_y}" class="{node.css_class}"/>'
            ),
            indent(text_block(node)),
            "</g>",
        )
    )


def node_svg(node: Node) -> str:
    if node.shape == "cylinder":
        return cylinder(node)
    return rectangle(node)


def line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    css_class: str,
    marker: str,
    *,
    bidirectional: bool = False,
) -> str:
    if x1 != x2 and y1 != y2:
        raise ValueError(f"non-orthogonal connector: ({x1}, {y1}) -> ({x2}, {y2})")
    marker_start = f' marker-start="url(#{marker})"' if bidirectional else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{css_class}"'
        f'{marker_start} marker-end="url(#{marker})"/>'
    )


def orthogonal_path(
    points: tuple[tuple[int, int], ...],
    css_class: str,
    marker: str,
    *,
    bidirectional: bool = False,
) -> str:
    if len(points) < 2:
        raise ValueError("an orthogonal path requires at least two points")

    commands = [f"M {points[0][0]} {points[0][1]}"]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 != x2 and y1 != y2:
            raise ValueError(f"non-orthogonal connector: ({x1}, {y1}) -> ({x2}, {y2})")
        commands.append(f"H {x2}" if y1 == y2 else f"V {y2}")

    marker_start = f' marker-start="url(#{marker})"' if bidirectional else ""
    return (
        f'<path d="{" ".join(commands)}" class="{css_class}"'
        f'{marker_start} marker-end="url(#{marker})"/>'
    )


def render_svg() -> str:
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="diagram-title diagram-desc">'
        ),
        "  <title id=\"diagram-title\">Tiny-LLM-Inference architecture</title>",
        (
            "  <desc id=\"diagram-desc\">A fixed-layout diagram showing LLMEngine input processing, "
            "EngineCore scheduling and model execution, the shared paged KV cache, and output processing.</desc>"
        ),
        "  <defs>",
        "    <marker id=\"arrow-call\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 Z\" fill=\"#334155\"/></marker>",
        "    <marker id=\"arrow-data\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 Z\" fill=\"#2563EB\"/></marker>",
        "    <marker id=\"arrow-storage\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 Z\" fill=\"#16A34A\"/></marker>",
        "  </defs>",
        "  <style>",
        "    .canvas-background { fill: #FFFFFF; }",
        "    text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #0F172A; }",
        "    .engine-group { fill: #F8FBFF; stroke: #60A5FA; stroke-width: 2; }",
        "    .core-group { fill: #FAF5FF; stroke: #A78BFA; stroke-width: 2; }",
        "    .scheduling-group { fill: #FCFAFF; stroke: #DDD6FE; stroke-width: 1.5; }",
        "    .execution-group { fill: #FFFBEB; stroke: #FED7AA; stroke-width: 1.5; }",
        "    .entry-node { fill: #EFF6FF; stroke: #2563EB; stroke-width: 2; }",
        "    .frontend-node { fill: #F8FAFC; stroke: #64748B; stroke-width: 2; }",
        "    .runtime-node { fill: #F5F3FF; stroke: #7C3AED; stroke-width: 2; }",
        "    .execution-node { fill: #FFF7ED; stroke: #EA580C; stroke-width: 2; }",
        "    .storage-node { fill: #F0FDF4; stroke: #16A34A; stroke-width: 2; }",
        "    .engine-title { fill: #1E3A8A; font-size: 21px; font-weight: 500; }",
        "    .engine-subtitle { fill: #1E3A8A; font-size: 16px; font-weight: 400; }",
        "    .core-title { fill: #334155; font-size: 18px; font-weight: 500; }",
        "    .core-subtitle, .stage-title, .lane-title { fill: #475569; font-size: 15px; font-weight: 400; }",
        "    .node-title { font-size: 17px; font-weight: 500; }",
        "    .node-subtitle { font-size: 14px; font-weight: 500; }",
        "    .call-flow, .data-flow, .storage-flow { fill: none; stroke-linecap: round; stroke-width: 2; }",
        "    .call-flow { stroke: #334155; }",
        "    .data-flow { stroke: #2563EB; stroke-dasharray: 7 6; }",
        "    .storage-flow { stroke: #16A34A; }",
        "  </style>",
        "",
        "  <!-- Keep the diagram readable when embedded on dark or colored pages. -->",
        f"  <rect width=\"{WIDTH}\" height=\"{HEIGHT}\" class=\"canvas-background\"/>",
        "",
        "  <!-- Ownership groups use fixed coordinates so edge changes never relayout the diagram. -->",
        "  <rect x=\"30\" y=\"125\" width=\"1460\" height=\"600\" rx=\"16\" class=\"engine-group\"/>",
        svg_text(760, 154, "LLMEngine", "engine-title"),
        svg_text(760, 177, "Text ↔ token boundary", "engine-subtitle"),
        "  <rect x=\"370\" y=\"190\" width=\"780\" height=\"495\" rx=\"14\" class=\"core-group\"/>",
        svg_text(760, 217, "EngineCore", "core-title"),
        svg_text(760, 239, "schedule → run → update", "core-subtitle"),
        "  <rect x=\"405\" y=\"250\" width=\"330\" height=\"400\" rx=\"12\" class=\"scheduling-group\"/>",
        svg_text(570, 276, "Scheduling", "stage-title"),
        "  <rect x=\"785\" y=\"250\" width=\"330\" height=\"400\" rx=\"12\" class=\"execution-group\"/>",
        svg_text(950, 276, "ModelRunner", "stage-title"),
        svg_text(190, 348, "Input processing", "lane-title"),
        svg_text(1330, 348, "Output processing", "lane-title"),
        "",
        "  <!-- Orthogonal runtime flow. Group-level arrows are mediated by EngineCore. -->",
        f"  {line(760, 100, 760, 125, 'call-flow', 'arrow-call')}",
        f"  {line(315, 406, 405, 406, 'data-flow', 'arrow-data')}",
        f"  {line(735, 406, 785, 406, 'data-flow', 'arrow-data', bidirectional=True)}",
        f"  {line(1115, 406, 1205, 406, 'data-flow', 'arrow-data')}",
        f"  {line(570, 353, 570, 385, 'call-flow', 'arrow-call')}",
        f"  {line(570, 453, 570, 500, 'storage-flow', 'arrow-storage')}",
        f"  {line(950, 353, 950, 385, 'call-flow', 'arrow-call')}",
        (
            "  "
            + orthogonal_path(
                ((900, 453), (900, 469), (872, 469), (872, 485)),
                "storage-flow",
                "arrow-storage",
                bidirectional=True,
            )
        ),
        (
            "  "
            + orthogonal_path(
                ((1000, 453), (1000, 469), (1028, 469), (1028, 485)),
                "data-flow",
                "arrow-data",
            )
        ),
        "",
        "  <!-- Executable components and the two views of one scheduler-owned KV cache. -->",
    ]

    for node in NODES:
        parts.append(indent(node_svg(node)))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing output differs from the generated SVG",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_svg()

    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}")
        print(f"{args.output} is up to date")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
