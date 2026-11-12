#!/bin/sh
# Rebuild data.json and index.html from results/. Usage: sh site/build.sh [repo-url]
set -e
cd "$(dirname "$0")"
python3 build_data.py
REPO="${1:-https://github.com/shyamvalsan/qwen-neuron-levers}" python3 - <<'PY'
import os
t = open("template.html").read()
d = open("data.json").read().replace("</", "<\\/").replace("\ufffd", "\\ufffd")
open("index.html", "w").write(t.replace("__DATA__", d).replace("__REPO__", os.environ["REPO"]))
print("index.html", round(os.path.getsize("index.html") / 1e6, 2), "MB")
PY
