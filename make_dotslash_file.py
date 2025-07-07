import argparse
from typing import Final
import urllib.request
import json
from dataclasses import dataclass


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
    # "windows-aarch64": PlatformConfig(...),
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


def fetch_latest_release() -> Release:
    with urllib.request.urlopen(
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


def platform_descriptor(platform: Platform, asset: Asset) -> object:
    extension = None
    for ext in ALLOWED_EXTENSIONS:
        if asset.browser_download_url.endswith(ext):
            extension = ext
    if extension is None:
        raise ValueError(
            f"Asset for {platform=} isn't supported by dotslash: {asset.browser_download_url}"
        )
    with urllib.request.urlopen(f"{asset.browser_download_url}.sha256") as response:
        if response.status != 200:
            raise RuntimeError(
                f"Failed to fetch digest for {asset.browser_download_url}: {response=}"
            )
        digest = bytes(response.read().strip()).decode()
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
