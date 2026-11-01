"""Command-line entry point for the stock research workflow."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown

from stock_research_workflow import build_graph


def run_streaming_research(question: str, console: Console) -> dict:
    graph = build_graph()
    result = {}
    stages = {
        "start_execution": "Starting research workflow",
        "begin_retrieval": "Starting parallel market and news retrieval",
        "prepare_report_input": "Preparing compact evidence for the report",
        "generate_report": "Generating the research report",
        "validate_report": "Validating report sections, figures, and citations",
        "correct_report": "Applying one targeted report correction",
        "finalize_execution": "Research workflow complete",
    }

    console.print("[cyan]Research progress[/cyan]")
    for update in graph.stream({"user_question": question}, stream_mode="updates"):
        for node, raw_update in update.items():
            node_update = raw_update or {}
            result.update(node_update)
            if node == "validate_input":
                if node_update.get("input_status") == "valid":
                    tickers = ", ".join(node_update.get("tickers", []))
                    console.print(
                        f"[green]✓[/green] Validated {tickers} "
                        f"(period: {node_update.get('period', 'default')})"
                    )
                else:
                    console.print("[red]✗[/red] Input validation failed")
                    for error in node_update.get("execution_errors", []):
                        console.print(f"  {error}")
                continue

            if node == "retrieve_market":
                available = ", ".join(node_update.get("market_summary", {}).keys()) or "none"
                console.print(f"[green]✓[/green] Market data retrieval finished (available: {available})")
                for error in node_update.get("market_errors", []):
                    console.print(f"  [yellow]Market data warning:[/yellow] {error}")
                continue

            if node == "retrieve_news":
                articles = len(node_update.get("news_evidence", []))
                console.print(f"[green]✓[/green] News retrieval finished ({articles} evidence items)")
                for error in node_update.get("news_errors", []):
                    console.print(f"  [yellow]News warning:[/yellow] {error}")
                continue

            if node == "validate_report":
                status = node_update.get("validation_status", "unknown")
                if status == "passed":
                    console.print("[green]✓[/green] Report validation passed")
                else:
                    console.print(f"[yellow]•[/yellow] Report validation: {status}")
                    for warning in node_update.get("validation_warnings", []):
                        console.print(f"  {warning}")
                continue

            label = stages.get(node)
            if label:
                console.print(f"[green]✓[/green] {label}")

    return result


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Market question: ")
    console = Console()
    try:
        result = run_streaming_research(question, console)
        report = result.get("report")
        if report:
            console.print(Markdown(report))
        else:
            console.print(f"[red]Research failed:[/red] {'; '.join(result.get('execution_errors', []))}")
        if result.get("validation_status") not in (None, "passed"):
            console.print(f"[yellow]Validation status:[/yellow] {result['validation_status']}")
        if result.get("validation_warnings"):
            console.print("[yellow]Validation warnings:[/yellow]")
            for warning in result["validation_warnings"]:
                console.print(f"- {warning}")
        if result.get("warnings"):
            console.print("[yellow]Data warnings:[/yellow]")
            for warning in result["warnings"]:
                console.print(f"- {warning}")
        if result.get("execution_errors") and result.get("report"):
            console.print("[yellow]Execution warnings:[/yellow]")
            for error in result["execution_errors"]:
                console.print(f"- {error}")
        if result.get("llm_usage"):
            console.print(f"LLM calls: {result['llm_usage'].get('calls', 0)}")
            console.print(
                "Token usage (estimated input / provider input / provider output): "
                f"{result['llm_usage'].get('estimated_input_tokens', 0)} / "
                f"{result['llm_usage'].get('actual_input_tokens', 0)} / "
                f"{result['llm_usage'].get('actual_output_tokens', 0)}"
            )
        if result.get("execution_time_seconds") is not None:
            console.print(f"Execution time: {result['execution_time_seconds']:.2f}s")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
