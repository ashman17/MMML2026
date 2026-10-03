#!/usr/bin/env python3
"""Two-stage atomic-claim and reasoning-dependency graph prototype."""

from __future__ import annotations

import argparse
import os
import sys
import time
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modal_client import (
    DEFAULT_MODAL_ENDPOINT,
    DEFAULT_MODAL_MODEL,
    ModalAPIError,
    generate_structured as modal_generate_structured,
)


DEFAULT_MODEL = DEFAULT_MODAL_MODEL
DEFAULT_CHECKPOINT = Path(__file__).with_name("artifacts") / "prompt1_checkpoint.json"
console = Console()
error_console = Console(stderr=True)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 524, 529}


class ClaimType(str, Enum):
    OBSERVATION = "OBSERVATION"
    ASSUMPTION = "ASSUMPTION"
    INFERENCE = "INFERENCE"
    ANSWER = "ANSWER"


class AtomicClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Sequential claim ID: C1, C2, C3, ...")
    claim: str = Field(description="One self-contained proposition")
    type: ClaimType
    source_text: str = Field(description="Exact source sentence or phrase")


class ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[AtomicClaim]


class Dependency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parents: list[str] = Field(
        description="Minimal joint set of parent claim IDs required by the child"
    )
    child: str = Field(description="Child claim ID")


class DependencyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edges: list[Dependency]


class Prompt1Checkpoint(BaseModel):
    version: int = 1
    model: str
    reasoning_trace: str
    claims: ClaimList


PROMPT_1 = """Convert the reasoning trace into atomic claims.

Rules:

- Each claim should contain one idea only.
- Do not correct or add information.
- Remove duplicates.
- Resolve pronouns so each claim is self-contained.

Types:

- OBSERVATION: directly stated visual evidence.
- ASSUMPTION: an unstated or weakly supported bridge used in the reasoning.
- INFERENCE: a conclusion derived from other claims.
- ANSWER: the final answer.

Return JSON:

{{
  "claims": [
    {{
      "id": "C1",
      "type": "OBSERVATION",
      "claim": "...",
      "source_text": "..."
    }}
  ]
}}

Reasoning trace:
<reasoning_trace>
{reasoning_trace}
</reasoning_trace>
"""


PROMPT_2 = """Build a directed reasoning graph from the claims.

NODE TYPES: OBSERVATION, ASSUMPTION, INFERENCE, ANSWER

ALLOWED EDGES:

OBSERVATION --> INFERENCE
ASSUMPTION --> INFERENCE
INFERENCE --> INFERENCE
OBSERVATION --> ANSWER
INFERENCE --> ANSWER

CONSTRAINTS:

- OBSERVATION has no parents.
- ANSWER has no children.
- Every INFERENCE has >= 1 parent.
- ANSWER has >= 1 parent.
- No cycles.
- Use only existing claim IDs.
- Add an edge only if the original reasoning uses the parent to support the child.
- Preserve incorrect reasoning; do not fix it.
- Prefer INFERENCE --> INFERENCE chains over shortcut edges.

OUTPUT:

{{
  "edges": [
    {{"parents": ["C1", "C2"], "child": "C5"}}
  ]
}}

Reasoning:
<reasoning_trace>
{reasoning_trace}
</reasoning_trace>

Claims:
<claims>
{claim_list}
</claims>
"""


def generate_structured(
    endpoint: str,
    token_id: str,
    token_secret: str,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    stage: str,
    attempts: int,
) -> BaseModel:
    for attempt in range(1, attempts + 1):
        try:
            return modal_generate_structured(
                endpoint=endpoint,
                token_id=token_id,
                token_secret=token_secret,
                model=model,
                prompt=prompt,
                schema=schema,
            )
        except ModalAPIError as exc:
            status_code = exc.status_code
            if status_code not in RETRYABLE_STATUS_CODES or attempt == attempts:
                raise
            delay = min(2**attempt, 10)
            error_console.print(
                f"[yellow]{stage}: Modal returned {status_code}; "
                f"retrying in {delay}s ({attempt}/{attempts})…[/yellow]\n"
                f"[dim]{exc}[/dim]"
            )
            time.sleep(delay)
    raise RuntimeError(f"{stage} failed after {attempts} attempts.")


def read_reasoning(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    if args.file is not None:
        return args.file.read_text(encoding="utf-8").strip()
    if sys.stdin.isatty():
        console.print("Paste the reasoning trace, then press [bold]Ctrl-D[/bold]:")
    return sys.stdin.read().strip()


def save_checkpoint(
    path: Path, *, model: str, reasoning_trace: str, claims: ClaimList
) -> None:
    checkpoint = Prompt1Checkpoint(
        model=model,
        reasoning_trace=reasoning_trace,
        claims=claims,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> Prompt1Checkpoint:
    if not path.exists():
        raise FileNotFoundError(f"Prompt 1 checkpoint not found: {path}")
    return Prompt1Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))


def render_claims(result: ClaimList) -> None:
    table = Table(title="Stage 1 — Atomic claims", title_style="bold cyan", show_lines=True)
    table.add_column("ID", style="bold yellow", no_wrap=True)
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Atomic claim", style="white", ratio=3)
    table.add_column("Source text", style="dim", ratio=2)
    for item in result.claims:
        table.add_row(
            item.id,
            item.type.value,
            item.claim,
            item.source_text,
        )
    console.print(table)


def render_graph(graph: DependencyGraph, claims: ClaimList) -> None:
    labels = {claim.id: claim.claim for claim in claims.claims}
    body = Text()
    if not graph.edges:
        body.append("No dependency edges were extracted.\n", style="dim")
    for dependency in graph.edges:
        parents = " + ".join(f"[{parent}]" for parent in dependency.parents)
        body.append(parents or "[no parents]", style="bold blue")
        body.append("  ──▶  ", style="bold white")
        body.append(f"[{dependency.child}]", style="bold yellow")
        child_label = labels.get(dependency.child)
        if child_label:
            body.append(f"  {child_label}", style="white")
        body.append("\n")
    console.print(Panel(body, title="Stage 2 — Dependency graph", border_style="cyan"))

    dependency_table = Table(title="Joint dependency sets", show_lines=True)
    dependency_table.add_column("Parents (joint set)", style="blue")
    dependency_table.add_column("Child", style="yellow")
    for dependency in graph.edges:
        dependency_table.add_row(", ".join(dependency.parents), dependency.child)
    console.print(dependency_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract claims and their reasoning dependency graph.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Reasoning trace supplied directly")
    source.add_argument("--file", type=Path, help="Read reasoning trace from a file")
    source.add_argument(
        "--resume",
        action="store_true",
        help="Skip Prompt 1 and load its trace and claims from --checkpoint",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--endpoint",
        help=f"Modal chat-completions endpoint (default: {DEFAULT_MODAL_ENDPOINT})",
    )
    parser.add_argument(
        "--prompt2-model",
        help="Optional model used only for dependency extraction (defaults to --model)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Total attempts for temporary Modal errors (default: 3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Prompt 1 checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Also print the validated JSON returned by both stages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        error_console.print("[red]Error: --attempts must be at least 1.[/red]")
        return 2
    claims: ClaimList | None = None
    prompt1_model = args.model
    if args.resume:
        try:
            checkpoint = load_checkpoint(args.checkpoint)
        except Exception as exc:
            error_console.print(f"[red]Error: {exc}[/red]")
            return 2
        reasoning = checkpoint.reasoning_trace
        claims = checkpoint.claims
        prompt1_model = checkpoint.model
        console.print(
            f"[green]Loaded Prompt 1 checkpoint:[/green] {args.checkpoint}\n"
            f"[dim]Saved Prompt 1 model: {checkpoint.model}[/dim]"
        )
    else:
        reasoning = read_reasoning(args)
        if not reasoning:
            error_console.print("[red]Error: no reasoning trace was provided.[/red]")
            return 2

    load_dotenv(Path(__file__).with_name(".env"), override=True)
    endpoint = args.endpoint or os.getenv("MODAL_ENDPOINT", DEFAULT_MODAL_ENDPOINT)
    token_id = os.getenv("MODAL_PROXY_TOKEN_ID")
    token_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not token_secret:
        error_console.print(
            "[red]Error: MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET "
            "must be set in .env.[/red]"
        )
        return 2

    try:
        if claims is None:
            console.rule("[bold cyan]Prompt 1: extracting atomic claims")
            claims = generate_structured(
                endpoint,
                token_id,
                token_secret,
                args.model,
                PROMPT_1.format(reasoning_trace=reasoning),
                ClaimList,
                stage="Prompt 1",
                attempts=args.attempts,
            )
            assert isinstance(claims, ClaimList)
            save_checkpoint(
                args.checkpoint,
                model=args.model,
                reasoning_trace=reasoning,
                claims=claims,
            )
            console.print(f"[green]Saved Prompt 1 checkpoint:[/green] {args.checkpoint}")
        else:
            console.rule("[bold cyan]Prompt 1: loaded from checkpoint")
        render_claims(claims)
        if args.raw_json:
            console.print_json(claims.model_dump_json(indent=2))

        console.rule("[bold cyan]Prompt 2: extracting dependencies")
        claim_json = claims.model_dump_json(indent=2)
        prompt2_model = args.prompt2_model or args.model
        if prompt2_model != prompt1_model:
            console.print(
                f"[dim]Prompt 2 model override: {prompt2_model} "
                f"(Prompt 1 used {prompt1_model})[/dim]"
            )
        graph = generate_structured(
            endpoint,
            token_id,
            token_secret,
            prompt2_model,
            PROMPT_2.format(reasoning_trace=reasoning, claim_list=claim_json),
            DependencyGraph,
            stage="Prompt 2",
            attempts=args.attempts,
        )
        assert isinstance(graph, DependencyGraph)
        render_graph(graph, claims)
        if args.raw_json:
            console.print_json(graph.model_dump_json(indent=2))
    except Exception as exc:
        error_console.print(f"[red]Error: {exc}[/red]")
        if (
            isinstance(exc, ModalAPIError)
            and exc.status_code in RETRYABLE_STATUS_CODES
        ):
            error_console.print(
                "[yellow]This is a temporary Modal model-serving failure. "
                "Retry or resume from the Prompt 1 checkpoint.[/yellow]"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
