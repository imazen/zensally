"""Streamed file identity without importing a tensor or model runtime."""
import hashlib


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
