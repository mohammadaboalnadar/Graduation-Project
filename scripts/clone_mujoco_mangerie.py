from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Unitree A1 MuJoCo model assets")
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO,
        help="Git repo URL for MuJoCo Menagerie",
    )
    parser.add_argument(
        "--target-dir",
        default="external/mujoco_menagerie",
        help="Where to clone the repository",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if target_dir.exists():
        print(f"Repository already exists at: {target_dir}")
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", args.repo_url, str(target_dir)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    model_xml = target_dir / "unitree_a1" / "scene.xml"
    if model_xml.exists():
        print(f"A1 scene file is ready: {model_xml}")
    else:
        print(
            "Clone succeeded, but expected A1 model file was not found. "
            "Check repository layout."
        )


if __name__ == "__main__":
    main()
