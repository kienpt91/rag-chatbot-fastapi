#!/bin/sh
# Runs the same evaluation set against a REAL provider, so that the numbers mean something about your model.
# This is the one command that spends money and sends the sample site's text and the dataset's questions to the API
# you configure. Put the settings in a file that is never committed (default eval/real.env; start from
# eval/real.env.example), then:
#
#     sh scripts/eval-real.sh                      # or: EVAL_ENV_FILE=path sh scripts/eval-real.sh --min-quality 0.9
#
# It runs in the same pinned container as everything else, with the network (it must reach the API) and nothing from
# your environment except that file. Extra arguments go to `rag-chatbot eval`.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
env_file=${EVAL_ENV_FILE:-$here/eval/real.env}
[ -f "$env_file" ] || { echo "no settings file: $env_file (copy eval/real.env.example and fill it in)" >&2; exit 2; }
ver=${PY_VERSION:-3.13}
case "$ver" in
  3.13) image=python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 ;; # same pins as scripts/py.sh
  3.12) image=python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 ;;
  *) echo "unsupported PY_VERSION (3.12 or 3.13)" >&2; exit 2 ;;
esac
mkdir -p "$here/out"
exec docker run --rm \
  --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges --cpus 2 --memory 2g --read-only \
  --tmpfs /tmp:rw,exec,size=256m -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/src \
  -e PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin --env-file "$env_file" \
  -v "$here:/work:ro" -v "$here/out:/work/out" -v "py-venv-$(basename "$here")-$ver:/venv:ro" -w /work \
  "$image" python -m rag_chatbot eval "$@"
