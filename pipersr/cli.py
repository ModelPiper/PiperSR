"""
PiperSR CLI — 2x super-resolution on Apple Neural Engine.

Usage:
    pipersr --input photo.png --output photo_2x.png
    pipersr -i photo.png

https://modelpiper.com | https://github.com/ModelPiper/PiperSR
"""

import argparse
from pathlib import Path

from pipersr import upscale


def main():
    parser = argparse.ArgumentParser(
        description="PiperSR — 2x super-resolution on Apple Neural Engine",
        epilog="Powered by PiperSR from ModelPiper — https://modelpiper.com",
    )
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_2x{p.suffix}")

    upscale(args.input, args.output)


if __name__ == "__main__":
    main()
