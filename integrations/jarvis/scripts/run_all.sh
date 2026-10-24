#!/bin/sh
set -eu
python -m jarvis_hpsafe.preflight
python -m jarvis_hpsafe.train
