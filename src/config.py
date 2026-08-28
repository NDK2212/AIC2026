
"""YAML + environment configuration loading.

The rule enforced here: every tunable comes from ``config/config.yaml`` and
every secret comes from the environment.  A malformed config raises
:class:`~src.schemas.ConfigError` at start-up rather than producing garbage
results later on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is optional at import time so ``--help`` always works
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False

from .schemas import ConfigError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require(node: dict[str, Any], key: str, where: str) -> Any:
    if key not in node:
        raise ConfigError(f"Missing required config key '{where}.{key}'")
    return node[key]


def _sub(node: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = node.get(key)
    if value is None:
        raise ConfigError(f"Missing required config section '{where}.{key}'")
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{where}.{key}' must be a mapping")
    return value


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


_DETECTED_DEVICE: str | None = None


def resolve_device(spec: str) -> str:
    """Turn ``"auto"`` into the best device actually available on this machine."""
    global _DETECTED_DEVICE
    if spec and spec != "auto":
        return spec
    if _DETECTED_DEVICE is not None:
        return _DETECTED_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DETECTED_DEVICE = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _DETECTED_DEVICE = "mps"
        else:
            _DETECTED_DEVICE = "cpu"
    except ImportError:
        _DETECTED_DEVICE = "cpu"
    return _DETECTED_DEVICE


# ---------------------------------------------------------------------------
# section dataclasses
# ---------------------------------------------------------------------------
@dataclass
class QdrantConfig:
    url: str
    collection: str
    vector_names: dict[str, str]
    payload: dict[str, str]
    prefetch_limit: int = 300
    search_limit: int = 200
    fusion: str = "rrf"
    timeout: int = 60
    api_key: str | None = None

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "QdrantConfig":
        vectors = _sub(node, "vector_names", "qdrant")
        payload = _sub(node, "payload", "qdrant")
        for name in ("siglip", "beit3"):
            _require(vectors, name, "qdrant.vector_names")
        for name in ("video_id", "frame_id"):
            _require(payload, name, "qdrant.payload")
        fusion = str(node.get("fusion", "rrf")).lower()
        if fusion not in {"rrf", "dbsf", "manual"}:
            raise ConfigError(
                f"qdrant.fusion must be one of rrf|dbsf|manual, got {fusion!r}"
            )
        return cls(
            url=str(_require(node, "url", "qdrant")),
            collection=str(_require(node, "collection", "qdrant")),
            vector_names={k: str(v) for k, v in vectors.items()},
            payload={k: str(v) for k, v in payload.items()},
            prefetch_limit=int(node.get("prefetch_limit", 300)),
            search_limit=int(node.get("search_limit", 200)),
            fusion=fusion,
            timeout=int(node.get("timeout", 60)),
            api_key=_env("QDRANT_API_KEY") or node.get("api_key"),
        )


@dataclass
class ElasticConfig:
    hosts: list[str]
    index: str
    fields: dict[str, str]
    size: int = 300
    min_score: float = 0.0
    timeout: int = 60
    verify_certs: bool = False
    boosts: dict[str, float] = field(default_factory=dict)
    phrase_slop: int = 3
    user: str | None = None
    password: str | None = None
    api_key: str | None = None

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "ElasticConfig":
        fields_node = _sub(node, "fields", "elasticsearch")
        for name in ("ocr", "asr", "video_id", "frame_id"):
            _require(fields_node, name, "elasticsearch.fields")
        hosts = _require(node, "hosts", "elasticsearch")
        if isinstance(hosts, str):
            hosts = [hosts]
        boosts = dict(node.get("boosts") or {})
        defaults = {"match": 2.0, "phrase": 3.0, "terms": 1.0, "exact_phrase": 6.0}
        defaults.update({k: float(v) for k, v in boosts.items()})
        return cls(
            hosts=[str(h) for h in hosts],
            index=str(_require(node, "index", "elasticsearch")),
            fields={k: str(v) for k, v in fields_node.items()},
            size=int(node.get("size", 300)),
            min_score=float(node.get("min_score", 0.0)),
            timeout=int(node.get("timeout", 60)),
            verify_certs=bool(node.get("verify_certs", False)),
            boosts=defaults,
            phrase_slop=int(node.get("phrase_slop", 3)),
            user=_env("ES_USER") or node.get("user") or node.get("username"),
            password=_env("ES_PASSWORD") or node.get("password"),
            api_key=_env("ES_API_KEY") or node.get("api_key"),
        )


@dataclass
class EncoderConfig:
    name: str
    enabled: bool
    backend: str
    model_id: str
    dim: int
    device: str
    max_length: int = 64
    batch_size: int = 16
    checkpoint_path: str | None = None
    tokenizer_path: str | None = None
    pretrained: str | None = None

    @classmethod
    def parse(cls, name: str, node: dict[str, Any]) -> "EncoderConfig":
        where = f"embedding.{name}"
        return cls(
            name=name,
            enabled=bool(node.get("enabled", True)),
            backend=str(node.get("backend", "transformers")),
            model_id=str(_require(node, "model_id", where)),
            dim=int(_require(node, "dim", where)),
            device=resolve_device(str(node.get("device", "auto"))),
            max_length=int(node.get("max_length", 64)),
            batch_size=int(node.get("batch_size", 16)),
            checkpoint_path=node.get("checkpoint_path"),
            tokenizer_path=node.get("tokenizer_path"),
            pretrained=node.get("pretrained"),
        )


@dataclass
class EmbeddingConfig:
    siglip: EncoderConfig
    beit3: EncoderConfig
    qwen: EncoderConfig | None = None

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "EmbeddingConfig":
        qwen_node = node.get("qwen")
        qwen_cfg = EncoderConfig.parse("qwen", qwen_node) if qwen_node else None
        return cls(
            siglip=EncoderConfig.parse("siglip", _sub(node, "siglip", "embedding")),
            beit3=EncoderConfig.parse("beit3", _sub(node, "beit3", "embedding")),
            qwen=qwen_cfg,
        )


@dataclass
class LLMConfig:
    provider: str = "nvidia"
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    base_url: str | None = None
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 4096
    enable_thinking: bool = True
    max_retries: int = 4
    retry_backoff: float = 2.0
    api_key_env: str = "NVIDIA_API_KEY"

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=str(node.get("provider", "nvidia")),
            model=str(_require(node, "model", "llm")),
            base_url=node.get("base_url"),
            temperature=float(node.get("temperature", 0.1)),
            top_p=float(node.get("top_p", 0.95)),
            max_tokens=int(node.get("max_tokens", 4096)),
            enable_thinking=bool(node.get("enable_thinking", True)),
            max_retries=int(node.get("max_retries", 4)),
            retry_backoff=float(node.get("retry_backoff", 2.0)),
        )


@dataclass
class VLMConfig:
    provider: str = "nvidia"
    model: str = ""
    base_url: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    max_workers: int = 4
    image_max_side: int = 768

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "VLMConfig":
        return cls(
            provider=str(node.get("provider", "nvidia")),
            model=str(_require(node, "model", "vlm")),
            base_url=node.get("base_url"),
            max_tokens=int(node.get("max_tokens", 256)),
            temperature=float(node.get("temperature", 0.0)),
            max_retries=int(node.get("max_retries", 3)),
            retry_backoff=float(node.get("retry_backoff", 2.0)),
            max_workers=int(node.get("max_workers", 4)),
            image_max_side=int(node.get("image_max_side", 768)),
        )


@dataclass
class FusionConfig:
    method: str = "weighted_rrf"
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    adaptive: bool = True
    adaptive_floor: float = 0.2

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "FusionConfig":
        method = str(node.get("method", "weighted_rrf")).lower()
        if method not in {"rrf", "weighted_rrf", "weighted_norm"}:
            raise ConfigError(
                "fusion.method must be rrf|weighted_rrf|weighted_norm, "
                f"got {method!r}"
            )
        weights = {k: float(v) for k, v in (node.get("weights") or {}).items()}
        for path in ("ocr", "asr", "description", "visual"):
            weights.setdefault(path, 1.0)
        return cls(
            method=method,
            rrf_k=int(node.get("rrf_k", 60)),
            weights=weights,
            adaptive=bool(node.get("adaptive", True)),
            adaptive_floor=float(node.get("adaptive_floor", 0.2)),
        )


@dataclass
class Blip2RerankConfig:
    enabled: bool = False
    model_id: str = "Salesforce/blip-itm-base-coco"
    device: str = "cpu"
    top_n: int = 25
    batch_size: int = 16
    weight: float = 1.0

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "Blip2RerankConfig":
        node = node or {}
        return cls(
            enabled=bool(node.get("enabled", False)),
            model_id=str(node.get("model_id", "Salesforce/blip-itm-base-coco")),
            device=resolve_device(str(node.get("device", "auto"))),
            top_n=int(node.get("top_n", 25)),
            batch_size=int(node.get("batch_size", 16)),
            weight=float(node.get("weight", 1.0)),
        )


@dataclass
class BGERerankConfig:
    enabled: bool = False
    model_id: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    top_n: int = 50
    weight: float = 0.8
    batch_size: int = 16

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "BGERerankConfig":
        node = node or {}
        return cls(
            enabled=bool(node.get("enabled", False)),
            model_id=str(node.get("model_id", "BAAI/bge-reranker-v2-m3")),
            device=resolve_device(str(node.get("device", "auto"))),
            top_n=int(node.get("top_n", 50)),
            weight=float(node.get("weight", 0.8)),
            batch_size=int(node.get("batch_size", 16)),
        )


@dataclass
class RerankConfig:
    blip2: Blip2RerankConfig = field(default_factory=Blip2RerankConfig)
    bge: BGERerankConfig = field(default_factory=BGERerankConfig)

    @property
    def enabled(self) -> bool:
        """True when either reranker is enabled."""
        return self.blip2.enabled or self.bge.enabled

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "RerankConfig":
        node = node or {}
        if "blip2" in node or "bge" in node:
            blip2 = Blip2RerankConfig.parse(node.get("blip2") or {})
            bge = BGERerankConfig.parse(node.get("bge") or {})
        else:
            blip2 = Blip2RerankConfig.parse(node)
            bge = BGERerankConfig(enabled=False)
        return cls(blip2=blip2, bge=bge)


@dataclass
class KeyframeConfig:
    root: Path
    map_dir: Path | None
    metadata_dir: Path | None
    image_glob: str = "*.jpg"

    @classmethod
    def parse(cls, node: dict[str, Any], base: Path) -> "KeyframeConfig":
        def _path(key: str) -> Path | None:
            raw = node.get(key)
            if not raw:
                return None
            p = Path(str(raw)).expanduser()
            return p if p.is_absolute() else (base / p).resolve()

        root = _path("root")
        if root is None:
            raise ConfigError("keyframes.root is required")
        return cls(
            root=root,
            map_dir=_path("map_dir"),
            metadata_dir=_path("metadata_dir"),
            image_glob=str(node.get("image_glob", "*.jpg")),
        )


@dataclass
class NeighborExpansionConfig:
    enabled: bool = True
    start_rank: int = 15
    offsets: list[int] = field(default_factory=lambda: [-15, 15, -30, 30])


@dataclass
class SubmissionConfig:
    max_rows: int = 100
    top_diverse: int = 8
    head_max_per_video: int = 3
    shot_window: int = 60
    neighbor_expansion: NeighborExpansionConfig = field(
        default_factory=NeighborExpansionConfig
    )

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "SubmissionConfig":
        ne_node = node.get("neighbor_expansion") or {}
        ne = NeighborExpansionConfig(
            enabled=bool(ne_node.get("enabled", True)),
            start_rank=int(ne_node.get("start_rank", 15)),
            offsets=[int(o) for o in (ne_node.get("offsets") or [-15, 15, -30, 30])],
        )
        return cls(
            max_rows=int(node.get("max_rows", 100)),
            top_diverse=int(node.get("top_diverse", 8)),
            head_max_per_video=int(node.get("head_max_per_video", 3)),
            shot_window=int(node.get("shot_window", 60)),
            neighbor_expansion=ne,
        )


@dataclass
class VQAConfig:
    vlm_top_n: int = 25
    llm_top_n: int = 25
    propagate: bool = True
    enrich_context: bool = True
    answer_max_chars: int = 100
    fallback_answer: str = "unknown"

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "VQAConfig":
        top_n = int(node.get("llm_top_n") or node.get("vlm_top_n", 25))
        return cls(
            vlm_top_n=top_n,
            llm_top_n=top_n,
            propagate=bool(node.get("propagate", True)),
            enrich_context=bool(node.get("enrich_context", True)),
            answer_max_chars=int(node.get("answer_max_chars", 100)),
            fallback_answer=str(node.get("fallback_answer", "unknown")),
        )


@dataclass
class TrakeConfig:
    anchor_index: int = 1
    per_step_topk: int = 150
    max_videos: int = 40
    paths_per_video: int = 3
    coverage_bonus: float = 0.5
    miss_penalty: float = -0.35
    min_gap: int = 1
    max_gap: int | None = None
    allow_fill: bool = True
    head_min_videos: int = 3
    head_window: int = 5
    step_weights: dict[str, float] = field(
        default_factory=lambda: {"visual": 0.6, "description": 0.4}
    )

    @classmethod
    def parse(cls, node: dict[str, Any]) -> "TrakeConfig":
        max_gap = node.get("max_gap")
        raw_weights = node.get("step_weights") or {}
        step_weights = {
            "visual": float(raw_weights.get("visual", 0.6)),
            "description": float(raw_weights.get("description", 0.4)),
        }
        return cls(
            anchor_index=int(node.get("anchor_index", 1)),
            per_step_topk=int(node.get("per_step_topk", 150)),
            max_videos=int(node.get("max_videos", 40)),
            paths_per_video=int(node.get("paths_per_video", 3)),
            coverage_bonus=float(node.get("coverage_bonus", 0.5)),
            miss_penalty=float(node.get("miss_penalty", -0.35)),
            min_gap=int(node.get("min_gap", 1)),
            max_gap=None if max_gap is None else int(max_gap),
            allow_fill=bool(node.get("allow_fill", True)),
            head_min_videos=int(node.get("head_min_videos", 3)),
            head_window=int(node.get("head_window", 5)),
            step_weights=step_weights,
        )


@dataclass
class MinioConfig:
    enabled: bool = True
    endpoint: str = "bucket.viettech.fit"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "aic2026"
    prefix: str = "keyframes/batch_1/"
    secure: bool = True
    max_retries: int = 5
    timeout: int = 30
    max_workers: int = 10
    cache_dir: Path = Path("./outputs/cache/keyframes")

    @classmethod
    def parse(cls, node: dict[str, Any], base: Path) -> "MinioConfig":
        def _path(raw: Any) -> Path:
            p = Path(str(raw)).expanduser()
            return p if p.is_absolute() else (base / p).resolve()

        sec_env = _env("MINIO_SECURE")
        secure = bool(node.get("secure", True)) if sec_env is None else (sec_env.lower() in ("true", "1", "yes"))

        return cls(
            enabled=bool(node.get("enabled", True)),
            endpoint=str(_env("MINIO_ENDPOINT") or node.get("endpoint", "bucket.viettech.fit")),
            access_key=str(_env("MINIO_ACCESS_KEY") or node.get("access_key", "minioadmin")),
            secret_key=str(_env("MINIO_SECRET_KEY") or node.get("secret_key", "minioadmin")),
            bucket=str(_env("MINIO_BUCKET") or node.get("bucket", "aic2026")),
            prefix=str(_env("MINIO_PREFIX") or node.get("prefix", "keyframes/batch_1/")),
            secure=secure,
            max_retries=int(node.get("max_retries", 5)),
            timeout=int(node.get("timeout", 30)),
            max_workers=int(node.get("max_workers", 10)),
            cache_dir=_path(node.get("cache_dir", "./outputs/cache/keyframes")),
        )


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: Path = Path("./outputs/cache")


@dataclass
class RunsConfig:
    dir: Path = Path("./outputs/runs")
    log_file: Path = Path("./outputs/runs/run.log")


# ---------------------------------------------------------------------------
# root config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """The fully parsed configuration tree."""

    path: Path
    root: Path
    qdrant: QdrantConfig
    elasticsearch: ElasticConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    vlm: VLMConfig
    fusion: FusionConfig
    rerank: RerankConfig
    keyframes: KeyframeConfig
    minio: MinioConfig
    submission: SubmissionConfig
    vqa: VQAConfig
    trake: TrakeConfig
    cache: CacheConfig
    runs: RunsConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path, *, no_cache: bool = False) -> "Config":
        """Read a YAML config, overlay environment secrets and validate it."""
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.is_file():
            raise ConfigError(f"Config file not found: {cfg_path}")

        # repo root = the directory that contains config/
        base = cfg_path.parent.parent if cfg_path.parent.name == "config" else cfg_path.parent
        load_dotenv(base / ".env", override=False)
        load_dotenv(override=False)

        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level")

        def _resolve(p: str | Path) -> Path:
            q = Path(str(p)).expanduser()
            return q if q.is_absolute() else (base / q).resolve()

        cache_node = raw.get("cache") or {}
        runs_node = raw.get("runs") or {}

        cfg = cls(
            path=cfg_path,
            root=base,
            qdrant=QdrantConfig.parse(_sub(raw, "qdrant", "<root>")),
            elasticsearch=ElasticConfig.parse(_sub(raw, "elasticsearch", "<root>")),
            embedding=EmbeddingConfig.parse(_sub(raw, "embedding", "<root>")),
            llm=LLMConfig.parse(_sub(raw, "llm", "<root>")),
            vlm=VLMConfig.parse(_sub(raw, "vlm", "<root>")),
            fusion=FusionConfig.parse(raw.get("fusion") or {}),
            rerank=RerankConfig.parse(raw.get("rerank") or {}),
            keyframes=KeyframeConfig.parse(_sub(raw, "keyframes", "<root>"), base),
            minio=MinioConfig.parse(raw.get("minio") or {}, base),
            submission=SubmissionConfig.parse(raw.get("submission") or {}),
            vqa=VQAConfig.parse(raw.get("vqa") or {}),
            trake=TrakeConfig.parse(raw.get("trake") or {}),
            cache=CacheConfig(
                enabled=bool(cache_node.get("enabled", True)) and not no_cache,
                dir=_resolve(cache_node.get("dir", "./outputs/cache")),
            ),
            runs=RunsConfig(
                dir=_resolve(runs_node.get("dir", "./outputs/runs")),
                log_file=_resolve(runs_node.get("log_file", "./outputs/runs/run.log")),
            ),
            raw=raw,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Cheap consistency checks that must hold before anything runs."""
        if self.submission.max_rows <= 0:
            raise ConfigError("submission.max_rows must be positive")
        if self.submission.top_diverse > self.submission.max_rows:
            raise ConfigError("submission.top_diverse cannot exceed submission.max_rows")
        if self.submission.shot_window < 0:
            raise ConfigError("submission.shot_window must be >= 0")
        if self.submission.head_max_per_video < 1:
            raise ConfigError("submission.head_max_per_video must be >= 1")
        if self.trake.min_gap < 0:
            raise ConfigError("trake.min_gap must be >= 0")
        if self.trake.max_gap is not None and self.trake.max_gap < self.trake.min_gap:
            raise ConfigError("trake.max_gap must be >= trake.min_gap")
        if self.trake.paths_per_video <= 0:
            raise ConfigError("trake.paths_per_video must be positive")
        if not (0.0 <= self.fusion.adaptive_floor <= 1.0):
            raise ConfigError("fusion.adaptive_floor must be within [0, 1]")
        if self.vqa.answer_max_chars <= 0:
            raise ConfigError("vqa.answer_max_chars must be positive")
        if not self.embedding.siglip.enabled and not self.embedding.beit3.enabled:
            raise ConfigError(
                "Both embedding.siglip.enabled and embedding.beit3.enabled are false - "
                "the visual retrieval path would have no encoder at all"
            )

    def ensure_dirs(self) -> None:
        """Create the output directories the pipeline writes into."""
        self.cache.dir.mkdir(parents=True, exist_ok=True)
        self.runs.dir.mkdir(parents=True, exist_ok=True)
        self.runs.log_file.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path = "config/config.yaml", *, no_cache: bool = False) -> Config:
    """Convenience wrapper around :meth:`Config.load`."""
    return Config.load(path, no_cache=no_cache)
