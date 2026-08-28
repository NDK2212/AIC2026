"""Command-line entrypoint: ``python -m src.cli <command>``.

Commands
--------
``inspect``    verify Qdrant / Elasticsearch / encoder wiring before a run
``decompose``  print the three sub-queries for one query (prompt debugging)
``search``     run retrieval only and show the fused top-k (fusion debugging)
``kis``        run one Textual KIS query into a CSV
``qa``         run one Q&A query into a CSV
``trake``      run one TRAKE query into a CSV
``batch``      run a whole query directory, dispatching on the filename suffix
``validate``   check a submission directory against the competition format
``pack``       validate then zip a submission directory
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Config, load_config
from .logging_utils import get_logger, setup_logging
from .schemas import PipelineError
from .utils.cache import DiskCache
from .utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)

TASK_SUFFIXES = ("kis", "qa", "trake")

# Global flags are declared with argparse.SUPPRESS so they can appear on either
# side of the subcommand; these are the values used when they appear on neither.
GLOBAL_DEFAULTS = {
    "config": "config/config.yaml",
    "no_cache": False,
    "verbose": False,
    "dry_run": False,
}


# ---------------------------------------------------------------------------
# lazy service container
# ---------------------------------------------------------------------------
@dataclass
class Services:
    """Lazily constructed clients, so ``validate``/``pack`` need no backends."""

    cfg: Config
    _cache: DiskCache | None = None
    _kf: KeyframeIndex | None = None
    _llm: Any = None
    _vlm: Any = None
    _qdrant: Any = None
    _elastic: Any = None
    _pipeline: Any = None

    @property
    def cache(self) -> DiskCache:
        if self._cache is None:
            self._cache = DiskCache(self.cfg.cache.dir, enabled=self.cfg.cache.enabled)
        return self._cache

    @property
    def keyframes(self) -> KeyframeIndex:
        if self._kf is None:
            self._kf = KeyframeIndex(
                root=self.cfg.keyframes.root,
                map_dir=self.cfg.keyframes.map_dir,
                metadata_dir=self.cfg.keyframes.metadata_dir,
                image_glob=self.cfg.keyframes.image_glob,
                cache=self.cache,
            )
        return self._kf

    @property
    def llm(self) -> Any:
        from .clients.llm import LLMClient

        if self._llm is None:
            self._llm = LLMClient(self.cfg.llm, self.cache)
        return self._llm

    @property
    def vlm(self) -> Any:
        from .clients.vlm import VLMClient

        if self._vlm is None:
            self._vlm = VLMClient(self.cfg.vlm, self.cache)
        return self._vlm

    @property
    def qdrant(self) -> Any:
        from .clients.qdrant import QdrantWrapper

        if self._qdrant is None:
            self._qdrant = QdrantWrapper(self.cfg.qdrant)
        return self._qdrant

    @property
    def elastic(self) -> Any:
        from .clients.elastic import ElasticWrapper

        if self._elastic is None:
            self._elastic = ElasticWrapper(self.cfg.elasticsearch)
        return self._elastic

    @property
    def pipeline(self) -> Any:
        from .retrieval.pipeline import RetrievalPipeline
        from .retrieval.rerank import BGEReranker, BLIP2Reranker
        from .retrieval.search_text import TextSearcher
        from .retrieval.search_visual import VisualSearcher

        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(
                cfg=self.cfg,
                llm=self.llm,
                text_searcher=TextSearcher(self.elastic),
                visual_searcher=VisualSearcher(self.cfg, self.qdrant, self.cache),
                blip2_reranker=BLIP2Reranker(self.cfg.rerank.blip2, self.keyframes),
                bge_reranker=BGEReranker(self.cfg.rerank.bge),
            )
        return self._pipeline


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_query(path: Path | str) -> str:
    """Read a BTC query file."""
    query_path = Path(path)
    if not query_path.is_file():
        raise PipelineError(f"Query file not found: {query_path}")
    text = query_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise PipelineError(f"Query file is empty: {query_path}")
    return text


def task_of(path: Path) -> str | None:
    """Infer the task from a query filename's suffix."""
    stem = path.stem.lower()
    for suffix in TASK_SUFFIXES:
        if stem.endswith(f"-{suffix}") or stem.endswith(f"_{suffix}"):
            return suffix
    return None


def default_out(cfg: Config, query_path: Path, out: str | None) -> Path:
    """Where a task's CSV goes when ``--out`` is not given."""
    if out:
        return Path(out)
    return cfg.root / "outputs" / "submission" / f"{query_path.stem}.csv"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace, services: Services) -> int:
    """Print the real backend schemas and assert the encoder dimensions match."""
    cfg = services.cfg
    failures: list[str] = []

    print("=" * 74)
    print(f"config      : {cfg.path}")
    print(f"repo root   : {cfg.root}")
    print("=" * 74)

    # -- Qdrant ------------------------------------------------------------
    print("\n[Qdrant]")
    print(f"  url        : {cfg.qdrant.url}")
    print(f"  collection : {cfg.qdrant.collection}")
    try:
        dims = services.qdrant.vector_dims()
        print(f"  vectors    : {dims or '<none>'}")
        count = services.qdrant.count()
        print(f"  points     : {count if count is not None else 'unknown'}")
        payloads = services.qdrant.sample_payload(1)
        if payloads:
            print(f"  payload    : {sorted(payloads[0])}")
            for role in ("video_id", "frame_id"):
                field = cfg.qdrant.payload[role]
                if field not in payloads[0]:
                    failures.append(
                        f"Qdrant payload has no field {field!r} "
                        f"(configured as qdrant.payload.{role})"
                    )
                else:
                    print(f"    {role:<9}: {field} = {payloads[0][field]!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Qdrant: {exc}")
        print(f"  ERROR      : {exc}")

    # -- Elasticsearch -----------------------------------------------------
    print("\n[Elasticsearch]")
    print(f"  hosts      : {cfg.elasticsearch.hosts}")
    print(f"  index      : {cfg.elasticsearch.index}")
    try:
        if not services.elastic.ping():
            failures.append("Elasticsearch did not answer a ping")
            print("  ERROR      : ping failed")
        else:
            fields = services.elastic.index_fields()
            print(f"  fields     : {fields}")
            print(f"  documents  : {services.elastic.count()}")
            roles_to_check = [r for r in ("ocr", "asr", "description", "video_id", "frame_id") if r in cfg.elasticsearch.fields]
            for role in roles_to_check:
                field = cfg.elasticsearch.fields[role]
                if field not in fields:
                    failures.append(
                        f"Elasticsearch index has no field {field!r} "
                        f"(configured as elasticsearch.fields.{role})"
                    )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Elasticsearch: {exc}")
        print(f"  ERROR      : {exc}")

    # -- encoders ----------------------------------------------------------
    print("\n[Text encoders]")
    if args.skip_encoders:
        print("  skipped (--skip-encoders)")
    else:
        from .retrieval.search_visual import VisualSearcher

        searcher = VisualSearcher(cfg, services.qdrant, services.cache)
        encoders = searcher.encoders()
        if not encoders:
            failures.append("No text encoder could be loaded for the visual path")
            print("  ERROR      : no encoder available")
        for vector_name, encoder in encoders.items():
            print(f"  {encoder.name:<8} -> vector {vector_name!r}, dim {encoder.dim}, "
                  f"device {encoder.cfg.device}")
        try:
            report = searcher.verify_dimensions()
            for vector_name, (enc_dim, col_dim) in report.items():
                print(f"  OK         : {vector_name} encoder {enc_dim}d == collection {col_dim}d")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Dimension check: {exc}")
            print(f"  ERROR      : {exc}")

    # -- keyframes ---------------------------------------------------------
    print("\n[Keyframes]")
    print(f"  root       : {cfg.keyframes.root}")
    kf = services.keyframes
    print(f"  videos     : {len(kf)}")
    if len(kf):
        sample = sorted(kf.videos)[0]
        frames = kf.all_frames(sample)
        print(f"  sample     : {sample} -> {len(frames)} keyframes, "
              f"frames {frames[0]}..{frames[-1]}, median gap {kf.median_gap(sample)}")
        exact = sum(1 for v in kf.videos.values() if v.exact)
        print(f"  mapped     : {exact}/{len(kf)} videos have a real frame-index map")
    else:
        print("  WARNING    : no keyframes found - Q&A and neighbour expansion are disabled")

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED with {len(failures)} problem(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All checks passed.")
    return 0


def cmd_decompose(args: argparse.Namespace, services: Services) -> int:
    """Print the JSON decomposition of a query."""
    from .retrieval.decompose import decompose

    query = args.query or read_query(args.query_file)
    result = decompose(
        services.llm,
        query,
        adaptive_floor=services.cfg.fusion.adaptive_floor,
        default_weights=services.cfg.fusion.weights,
        use_cache=services.cfg.cache.enabled,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace, services: Services) -> int:
    """Run retrieval only and print the fused top-k with per-path detail."""
    query = args.query or read_query(args.query_file)
    candidates, result = services.pipeline.run(
        query, topk=args.topk, use_cache=services.cfg.cache.enabled
    )

    print("\nDecomposition")
    print("-" * 74)
    for label, value in (
        ("ocr", result.ocr_query),
        ("asr", result.asr_query),
        ("desc", result.description_query),
        ("image", result.image_query),
    ):
        print(f"  {label:<6}: {value}")
    print(f"  weights: {result.modality_weights}")

    print(f"\nTop {min(args.topk, len(candidates))} of {len(candidates)} fused candidates")
    print("-" * 74)
    print(f"{'#':>3}  {'video_id':<14} {'frame':>8}  {'score':>9}  per-path rank")
    for candidate in candidates[: args.topk]:
        ranks = candidate.extra.get("per_path_rank", {})
        detail = ", ".join(f"{p}#{r}" for p, r in sorted(ranks.items()))
        print(f"{candidate.rank:>3}  {candidate.video_id:<14} {candidate.frame_id:>8}  "
              f"{candidate.score:>9.6f}  {detail}")
    return 0


def cmd_kis(args: argparse.Namespace, services: Services) -> int:
    """Run one Textual KIS query."""
    from .submission.writer import write_kis
    from .tasks.kis import run_kis

    query_path = Path(args.query_file)
    query = read_query(query_path)
    rows = run_kis(
        query, services.pipeline, services.keyframes, services.cfg,
        trace_name=query_path.stem, use_cache=services.cfg.cache.enabled,
    )
    out = default_out(services.cfg, query_path, args.out)
    if args.dry_run:
        _preview(rows, out)
        return 0
    write_kis(out, rows, max_rows=services.cfg.submission.max_rows)
    print(f"Wrote {min(len(rows), services.cfg.submission.max_rows)} rows to {out}")
    return 0


def cmd_qa(args: argparse.Namespace, services: Services) -> int:
    """Run one Q&A query."""
    from .submission.writer import write_qa
    from .tasks.vqa import run_vqa

    query_path = Path(args.query_file)
    query = read_query(query_path)
    rows = run_vqa(
        query, services.pipeline, services.keyframes, services.cfg,
        trace_name=query_path.stem, use_cache=services.cfg.cache.enabled,
    )
    out = default_out(services.cfg, query_path, args.out)
    if args.dry_run:
        _preview(rows, out)
        return 0
    write_qa(
        out, rows,
        max_rows=services.cfg.submission.max_rows,
        answer_max_chars=services.cfg.vqa.answer_max_chars,
        fallback_answer=services.cfg.vqa.fallback_answer,
    )
    print(f"Wrote {min(len(rows), services.cfg.submission.max_rows)} rows to {out}")
    return 0


def cmd_trake(args: argparse.Namespace, services: Services) -> int:
    """Run one TRAKE query."""
    from .submission.writer import write_trake
    from .tasks.trake import run_trake, save_plan

    query_path = Path(args.query_file)
    query = read_query(query_path)
    rows, plan = run_trake(
        query, services.pipeline, services.keyframes, services.cfg,
        trace_name=query_path.stem, use_cache=services.cfg.cache.enabled,
    )
    out = default_out(services.cfg, query_path, args.out)
    save_plan(services.cfg, plan, out.stem)
    if args.dry_run:
        _preview(rows, out)
        return 0
    write_trake(out, rows, num_events=plan.num_events,
                max_rows=services.cfg.submission.max_rows)
    print(f"Wrote {min(len(rows), services.cfg.submission.max_rows)} rows "
          f"({plan.num_events} events each) to {out}")
    return 0


def cmd_batch(args: argparse.Namespace, services: Services) -> int:
    """Run every query file in a directory, continuing past individual failures."""
    from .submission.writer import write_kis, write_qa, write_trake
    from .tasks.kis import run_kis
    from .tasks.trake import run_trake, save_plan
    from .tasks.vqa import run_vqa

    query_dir = Path(args.query_dir)
    if not query_dir.is_dir():
        raise PipelineError(f"Query directory not found: {query_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else services.cfg.root / "outputs" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(query_dir.glob("*.txt"))
    if not files:
        raise PipelineError(f"No .txt query files found in {query_dir}")

    try:
        from tqdm import tqdm

        iterator = tqdm(files, desc="queries", unit="q")
    except ImportError:  # pragma: no cover
        iterator = files

    cfg = services.cfg
    results: list[tuple[str, str, str]] = []   # (file, task, status)

    for query_path in iterator:
        task = task_of(query_path)
        if task is None:
            log.warning("Skipping %s - no -kis / -qa / -trake suffix", query_path.name)
            results.append((query_path.name, "?", "SKIPPED (unknown task)"))
            continue

        out = out_dir / f"{query_path.stem}.csv"
        try:
            query = read_query(query_path)
            if task == "kis":
                rows = run_kis(query, services.pipeline, services.keyframes, cfg,
                               trace_name=query_path.stem, use_cache=cfg.cache.enabled)
                written = 0 if args.dry_run else write_kis(out, rows, cfg.submission.max_rows)
            elif task == "qa":
                rows = run_vqa(query, services.pipeline, services.keyframes,
                               cfg, trace_name=query_path.stem, use_cache=cfg.cache.enabled)
                written = 0 if args.dry_run else write_qa(
                    out, rows, cfg.submission.max_rows,
                    cfg.vqa.answer_max_chars, cfg.vqa.fallback_answer,
                )
            else:
                rows, plan = run_trake(query, services.pipeline, services.keyframes, cfg,
                                       trace_name=query_path.stem,
                                       use_cache=cfg.cache.enabled)
                save_plan(cfg, plan, out.stem)
                written = 0 if args.dry_run else write_trake(
                    out, rows, plan.num_events, cfg.submission.max_rows
                )
            count = len(rows) if args.dry_run else written
            results.append((query_path.name, task, f"OK ({count} rows)"))
        except Exception as exc:  # noqa: BLE001 - one query must not stop the batch
            log.error("%s failed: %s", query_path.name, exc, exc_info=True)
            results.append((query_path.name, task, f"FAILED: {exc}"))

    print("\n" + "=" * 74)
    print(f"{'query file':<32} {'task':<7} status")
    print("-" * 74)
    for name, task, status in results:
        print(f"{name:<32} {task:<7} {status}")
    print("=" * 74)

    failed = sum(1 for _, _, status in results if status.startswith("FAILED"))
    print(f"{len(results) - failed}/{len(results)} queries succeeded")
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace, services: Services) -> int:
    """Check a submission directory against the competition format."""
    from .submission.validator import load_expected_events, validate

    directory = Path(args.dir) if args.dir else services.cfg.root / "outputs" / "submission"
    expected = load_expected_events(services.cfg.runs.dir)
    errors = validate(directory, expected)

    if errors:
        print(f"{len(errors)} problem(s) found in {directory}:")
        for error in errors:
            print(f"  - {error}")
        return 1
    files = sorted(directory.glob("*.csv"))
    print(f"{directory}: {len(files)} file(s) valid")
    for path in files:
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  - {path.name}: {rows} rows")
    return 0


def cmd_pack(args: argparse.Namespace, services: Services) -> int:
    """Validate then zip a submission directory."""
    from .submission.packer import pack, verify_zip
    from .submission.validator import load_expected_events

    directory = Path(args.dir) if args.dir else services.cfg.root / "outputs" / "submission"
    expected = load_expected_events(services.cfg.runs.dir)
    target = pack(directory, args.out, expected, skip_validation=args.no_validate)

    problems = verify_zip(target)
    if problems:
        print("Archive verification failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Packed {target} ({target.stat().st_size / 1024:.1f} KiB)")
    print("Archive layout: submission/<query>.csv")
    return 0


def _preview(rows: Sequence[Any], out: Path) -> None:
    """Print what would be written under ``--dry-run``."""
    print(f"[dry-run] would write {len(rows)} rows to {out}")
    for row in rows[:10]:
        print("   ", ",".join(str(x) for x in (row if isinstance(row, (list, tuple)) else [row])))
    if len(rows) > 10:
        print(f"    ... and {len(rows) - 10} more")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every subcommand."""
    # Global flags live on a parent parser so they work on either side of the
    # subcommand.  SUPPRESS keeps an unspecified subparser flag from clobbering
    # a value already given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="path to config.yaml")
    common.add_argument("--no-cache", action="store_true", default=argparse.SUPPRESS,
                        help="bypass the LLM/embedding cache")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="debug logging")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="compute but write no CSV")

    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="AIC 2026 online video-retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[common],
    )
    # NB: no parser.set_defaults() for the global flags here - it mutates
    # ``action.default`` on the *shared* parent actions, replacing SUPPRESS and
    # letting the subparser clobber a value given before the subcommand.
    # apply_global_defaults() fills them in after parsing instead.

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", parents=[common],
                       help="verify backends, schemas and encoder dimensions")
    p.add_argument("--skip-encoders", action="store_true",
                   help="do not load the embedding models")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("decompose", parents=[common], help="print the OCR/ASR/IMAGE decomposition")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="query text")
    group.add_argument("--query-file", help="path to a query .txt file")
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("search", parents=[common], help="run retrieval only and print the fused top-k")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="query text")
    group.add_argument("--query-file", help="path to a query .txt file")
    p.add_argument("--topk", type=int, default=20, help="rows to print (default 20)")
    p.set_defaults(func=cmd_search)

    for name, func, help_text in (
        ("kis", cmd_kis, "run one Textual KIS query"),
        ("qa", cmd_qa, "run one Q&A query"),
        ("trake", cmd_trake, "run one TRAKE query"),
    ):
        p = sub.add_parser(name, parents=[common], help=help_text)
        p.add_argument("--query-file", required=True, help="path to the query .txt file")
        p.add_argument("--out", help="output CSV path")
        p.set_defaults(func=func)

    p = sub.add_parser("batch", parents=[common], help="run a whole query directory")
    p.add_argument("--query-dir", required=True, help="directory of query .txt files")
    p.add_argument("--out-dir", help="directory for the result CSVs")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("validate", parents=[common], help="check a submission directory")
    p.add_argument("--dir", help="submission directory")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("pack", parents=[common], help="validate then zip a submission directory")
    p.add_argument("--dir", help="submission directory")
    p.add_argument("--out", required=True, help="output .zip path")
    p.add_argument("--no-validate", action="store_true",
                   help="package even if validation fails (not recommended)")
    p.set_defaults(func=cmd_pack)

    return parser


def apply_global_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any global flag that appeared on neither side of the subcommand."""
    for name, default in GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, honouring global flags in either position."""
    return apply_global_defaults(build_parser().parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint.  Returns the process exit code."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = parse_args(argv)

    try:
        cfg = load_config(args.config, no_cache=args.no_cache)
    except Exception as exc:  # noqa: BLE001
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    cfg.ensure_dirs()
    setup_logging(verbose=args.verbose, log_file=cfg.runs.log_file)
    log.debug("Loaded config from %s", cfg.path)

    services = Services(cfg=cfg)
    func: Callable[[argparse.Namespace, Services], int] = args.func
    try:
        return func(args, services)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        else:
            print("Re-run with -v for a full traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
