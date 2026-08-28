"""Thin wrappers around the external services: Qdrant, Elasticsearch, LLM, VLM, MinIO."""

from .minio_client import MinioKeyframeClient

__all__ = ["MinioKeyframeClient"]
