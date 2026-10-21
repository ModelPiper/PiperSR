#!/usr/bin/env python3
"""
PiperSR — Real-time 2x super-resolution on Apple Neural Engine.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/ModelPiper/PiperSR

Runs the bundled model straight from a clone. Same as the `pipersr` CLI:
    python inference.py -i photo.png -o photo_2x.png

License: AGPL-3.0 (code), CC BY 4.0 (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

from pipersr.cli import main


if __name__ == "__main__":
    main()
