"""CLI entry point: ``negmas-mcp`` / ``python -m negmas_llm.mcp``."""

from __future__ import annotations

import argparse
from typing import Any

from negmas_llm.mcp.server import make_server

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Run the negmas MCP server."""
    parser = argparse.ArgumentParser(
        prog="negmas-mcp", description="Expose negmas negotiators over MCP."
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="transport to serve on (default: stdio)",
    )
    parser.add_argument("--host", default=None, help="bind host for HTTP transports")
    parser.add_argument(
        "--port", type=int, default=None, help="bind port for HTTP transports"
    )
    parser.add_argument("--name", default="negmas", help="server name")
    args = parser.parse_args(argv)

    server = make_server(args.name)
    kwargs: dict[str, Any] = {}
    if args.host is not None:
        kwargs["host"] = args.host
    if args.port is not None:
        kwargs["port"] = args.port
    server.run(transport=args.transport, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    main()
