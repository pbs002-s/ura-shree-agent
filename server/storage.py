"""
Where snapshot blobs live.

The Time Machine is content-addressed, which makes the storage question easy:
a blob is immutable and named by its own sha256, so it can sit anywhere that
can hold bytes under a key. On a laptop that is `.timemachine/objects/`. In a
cloud deployment it has to be object storage, because the API runs on several
replicas and a blob written by one of them has to be readable by all the
others - a local directory silently gives each replica its own private,
incomplete history.

Both implementations are write-once: a key that already exists is never
rewritten. That is not an optimisation, it is what makes concurrent writers
safe without a lock.
"""

from __future__ import annotations

import os
import threading
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse


class ObjectStore(ABC):
    """A flat, immutable key to bytes mapping."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Stores `data` under `key`, doing nothing if the key already exists."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Returns the stored bytes, or None when the key is absent."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def keys(self, prefix: str = "") -> Iterator[str]:
        ...

    @abstractmethod
    def total_bytes(self, prefix: str = "") -> int:
        ...


class LocalObjectStore(ObjectStore):
    """
    A sharded directory of zlib-compressed blobs.

    Content is written to a temporary name and renamed into place, so a crash
    mid-write cannot leave a truncated object that a later read would trust as
    valid content.
    """

    def __init__(self, root: str, compress_level: int = 6):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.compress_level = compress_level
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        # Two-character shard: a flat directory of a hundred thousand files is
        # slow to list on every filesystem that matters.
        safe = key.replace("/", "_")
        return self.root / safe[:2] / safe

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(zlib.compress(data, self.compress_level))
        with self._lock:
            tmp.replace(path)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return zlib.decompress(path.read_bytes())
        except (OSError, zlib.error):
            return None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def keys(self, prefix: str = "") -> Iterator[str]:
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp") and path.name.startswith(prefix):
                yield path.name

    def total_bytes(self, prefix: str = "") -> int:
        return sum(
            p.stat().st_size
            for p in self.root.rglob("*")
            if p.is_file() and p.name.startswith(prefix)
        )


class S3ObjectStore(ObjectStore):
    """
    S3, or anything that speaks its API: MinIO, R2, GCS in interop mode.

    `boto3` is imported lazily so a local install never needs it. Blobs are
    compressed before upload for the same reason as locally - source trees
    compress by roughly four to one, and storage is billed by the byte.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "timemachine/",
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        compress_level: int = 6,
    ):
        try:
            import boto3  # noqa: PLC0415  - optional dependency, cloud only
        except ImportError as err:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Object storage needs boto3. Install the cloud extras: "
                "pip install -r requirements-cloud.txt"
            ) from err

        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.compress_level = compress_level
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key[:2]}/{key}"

    def put(self, key: str, data: bytes) -> None:
        if self.exists(key):
            return
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=zlib.compress(data, self.compress_level),
        )

    def get(self, key: str) -> Optional[bytes]:
        try:
            body = self._client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception:
            # A missing object and an unreachable bucket both mean "no content
            # here"; the caller reports the file as unrecoverable either way.
            return None
        try:
            return zlib.decompress(body)
        except zlib.error:
            return body

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def keys(self, prefix: str = "") -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                name = item["Key"].rsplit("/", 1)[-1]
                if name.startswith(prefix):
                    yield name

    def total_bytes(self, prefix: str = "") -> int:
        total = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                if item["Key"].rsplit("/", 1)[-1].startswith(prefix):
                    total += item["Size"]
        return total


def build_object_store(url: str, fallback_dir: str, **options) -> ObjectStore:
    """
    Picks a store from a URL.

      s3://bucket/prefix        S3, MinIO or any S3-compatible endpoint
      file:///var/lib/objects   an explicit local directory
      (empty)                   `fallback_dir`, the single-machine default
    """
    if not url:
        return LocalObjectStore(fallback_dir)

    parsed = urlparse(url)
    if parsed.scheme in ("s3", "minio", "gs"):
        return S3ObjectStore(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/") or "timemachine/",
            **options,
        )
    if parsed.scheme in ("", "file"):
        return LocalObjectStore(parsed.path or url)
    raise ValueError(f"Unsupported object store URL: {url}")
