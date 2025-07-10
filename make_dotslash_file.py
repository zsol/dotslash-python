from __future__ import annotations

import argparse
import json
import os
from urllib.parse import unquote, urljoin
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import _TemporaryFileWrapper
from typing import Final, Generator, Protocol, Any
import atexit
import tempfile


@dataclass(frozen=True)
class Asset:
    name: str
    browser_download_url: str
    state: str
    size: int


@dataclass(frozen=True)
class Release:
    name: str
    tag_name: str
    draft: bool
    prerelease: bool
    assets: list[Asset]


@dataclass(frozen=True)
class PlatformConfig:
    marker: str
    flavor: str
    path: str


@dataclass(frozen=True)
class Platform:
    name: str
    free_threaded: bool = False


PLATFORMS: Final[dict[Platform, PlatformConfig]] = {
    Platform("linux-aarch64"): PlatformConfig(
        marker="aarch64-unknown-linux-gnu",
        flavor="install_only_stripped",
        path="python/bin/python",
    ),
    Platform("linux-aarch64", free_threaded=True): PlatformConfig(
        marker="aarch64-unknown-linux-gnu",
        flavor="freethreaded+pgo+lto-full",
        path="python/install/bin/python",
    ),
    Platform("linux-x86_64"): PlatformConfig(
        marker="x86_64_v3-unknown-linux-gnu",
        flavor="install_only_stripped",
        path="python/bin/python",
    ),
    Platform("linux-x86_64", free_threaded=True): PlatformConfig(
        marker="x86_64_v3-unknown-linux-gnu",
        flavor="freethreaded+pgo+lto-full",
        path="python/install/bin/python",
    ),
    Platform("macos-aarch64"): PlatformConfig(
        marker="aarch64-apple-darwin",
        flavor="install_only_stripped",
        path="python/bin/python",
    ),
    Platform("macos-aarch64", free_threaded=True): PlatformConfig(
        marker="aarch64-apple-darwin",
        flavor="freethreaded+pgo+lto-full",
        path="python/install/bin/python",
    ),
    Platform("macos-x86_64"): PlatformConfig(
        marker="x86_64-apple-darwin",
        flavor="install_only_stripped",
        path="python/bin/python",
    ),
    Platform("macos-x86_64", free_threaded=True): PlatformConfig(
        marker="x86_64-apple-darwin",
        flavor="freethreaded+pgo+lto-full",
        path="python/install/bin/python",
    ),
    Platform("windows-aarch64"): PlatformConfig(
        marker="aarch64-pc-windows-msvc",
        flavor="install_only_stripped",
        path="python/python.exe",
    ),
    Platform("windows-aarch64", free_threaded=True): PlatformConfig(
        marker="aarch64-pc-windows-msvc",
        flavor="freethreaded+pgo-full",
        path="python/install/python.exe",
    ),
    Platform("windows-x86_64"): PlatformConfig(
        marker="x86_64-pc-windows-msvc",
        flavor="install_only_stripped",
        path="python/python.exe",
    ),
    Platform("windows-x86_64", free_threaded=True): PlatformConfig(
        marker="x86_64-pc-windows-msvc",
        flavor="freethreaded+pgo-full",
        path="python/install/python.exe",
    ),
}


class _Response(Protocol):
    def info(self) -> dict[str, str]: ...

    @property
    def status(self) -> int | None: ...


class Response(_Response, _TemporaryFileWrapper):
    pass


@contextmanager
def request(url: str) -> Generator[Response, None, None]:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        yield response


def fetch_latest_release() -> Release:
    with request(
        "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to fetch release info: {response.status}")
        release_data = json.loads(response.read())
        return Release(
            name=release_data["name"],
            tag_name=release_data["tag_name"],
            draft=release_data["draft"],
            prerelease=release_data["prerelease"],
            assets=[
                Asset(
                    name=asset["name"],
                    browser_download_url=asset["browser_download_url"],
                    state=asset["state"],
                    size=asset["size"],
                )
                for asset in release_data["assets"]
            ],
        )


def find_asset_for_platform(
    release: Release, version: str, platform: Platform
) -> Asset:
    ret: list[Asset] = []
    platform_cfg = PLATFORMS[platform]
    for asset in release.assets:
        if not asset.name.startswith(f"cpython-{version}."):
            continue
        if asset.name.endswith(".sha256"):
            continue
        if platform_cfg.marker in asset.name and platform_cfg.flavor in asset.name:
            ret.append(asset)
    if len(ret) > 1:
        raise ValueError(
            f"More than one asset matches {platform_cfg} for {version=}, {platform=}. Candidates: {[a.name for a in ret]}"
        )
    if len(ret) == 0:
        raise ValueError(
            f"No assets found for {version=}, {platform=} in {release.name=}"
        )
    return ret[0]


ALLOWED_EXTENSIONS = ["tar.gz", "tar.zst"]


# Module-level cache for SHA256SUMS: {release_tag: {filename: digest}}
_sha256sums_cache: dict[str, dict[str, str]] = {}


def _fetch_and_cache_sha256sums(asset_url: str) -> dict[str, str]:
    sha256sums_url = urljoin(asset_url, "SHA256SUMS")
    release_tag = unquote(sha256sums_url.rsplit("/", 2)[-2])  # e.g., 20250708
    if release_tag in _sha256sums_cache:
        return _sha256sums_cache[release_tag]
    with request(sha256sums_url) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Failed to fetch SHA256SUMS: {sha256sums_url} {response=}"
            )
        data: str = response.read().decode()
    digest_map: dict[str, str] = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            digest, filename = parts
            digest_map[filename] = digest
    _sha256sums_cache[release_tag] = digest_map
    return digest_map


def fetch_sha256_digest(browser_download_url: str) -> str:
    digest_map = _fetch_and_cache_sha256sums(browser_download_url)
    filename = unquote(browser_download_url.rsplit("/", 1)[-1])
    if filename in digest_map:
        return digest_map[filename]
    raise RuntimeError(f"Digest for {filename} not found in SHA256SUMS")


def platform_descriptor(platform: Platform, asset: Asset) -> object:
    extension = None
    for ext in ALLOWED_EXTENSIONS:
        if asset.browser_download_url.endswith(ext):
            extension = ext
    if extension is None:
        raise ValueError(
            f"Asset for {platform=} isn't supported by dotslash: {asset.browser_download_url}"
        )

    digest = fetch_sha256_digest(asset.browser_download_url)
    return {
        "size": asset.size,
        "hash": "sha256",
        "digest": digest,
        "format": extension,
        "path": PLATFORMS[platform].path,
        "providers": [{"url": asset.browser_download_url}],
        # this is needed on linux/macos so the interpreter can locate the stdlib and
        # other runtime files; it's ignored on windows
        "arg0": "underlying-executable",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpython-version", default="3.13")
    parser.add_argument(
        "--free-threaded", action="store_true", help="Look for free-threaded builds"
    )
    args = parser.parse_args()
    version = args.cpython_version
    assert isinstance(version, str)
    rel = fetch_latest_release()
    platform_descriptors = {
        platform.name: platform_descriptor(
            platform, find_asset_for_platform(rel, version, platform)
        )
        for platform in PLATFORMS.keys()
        if platform.free_threaded == args.free_threaded
        # windows-aarch64 builds start at 3.11
        if not (platform.name == "windows-aarch64" and version in {"3.9", "3.10"})
    }
    version_suffix = "t" if args.free_threaded else ""
    descriptor = {
        "name": f"cpython-{version}{version_suffix}",
        "platforms": platform_descriptors,
    }
    print("#!/usr/bin/env dotslash")
    print()
    print(json.dumps(descriptor, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
