#!/usr/bin/env bash
# warning: needs 12GiB of free ram
set -e
python3 -m coverage run -m unittest
PYTHONPATH=. python3 -m coverage run --append docs/gen_images.py --force
python3 -m coverage report
python3 -m coverage html
