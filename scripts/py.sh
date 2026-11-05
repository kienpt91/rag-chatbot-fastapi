#!/bin/sh
# Run a command inside a throwaway Python container with the project mounted READ-ONLY.
# The only writable places are /work/out (a bind of ./out: coverage data, caches, dist/), /tmp (tmpfs) and the
# venv volume. The container has no access to $HOME or host env, drops all capabilities, runs as a non-root
# user, has no network except for `install`, and is limited to 4 CPUs and 6 GiB.
#
# usage: scripts/py.sh install            create the venv and install requirements.txt + requirements-dev.txt (--require-hashes)
#        scripts/py.sh <shell command>    run a command with the venv on PATH, no network
#   PY_VERSION=3.13 (default) or 3.12
#   PY_RW=1          mount the source read-write (only for `ruff format` and `ruff check --fix`)
#   PY_CPUS=2        CPU limit of the container (default 4); the CI runners have 2
#   PY_NET=bridge    allow the network for one command (only scripts/lock-deps.py and scripts/check-dep-age.py need it)
#   PY_DB=1          start a throwaway Postgres + pgvector container (pinned by digest, below) and run the command in
#                    a container that shares its network namespace: the database is at 127.0.0.1:5432 and
#                    $RAG_TEST_DATABASE_URL names it (superuser; the random password is in $PGPASSWORD, which libpq reads; gone when
#                    the command ends).
#                    Neither container has a route out: the database sits on an --internal docker network.
#
# Images are the official `python` slim-trixie images and the official pgvector image, pinned by the digest of the
# multi-arch index:
#   python 3.13.17-slim-trixie     index pushed 2026-10-24
#   python 3.12.15-slim-trixie     index pushed 2026-10-22
#   pgvector 0.8.6-pg16            image created 2026-08-13 (the tag is a Postgres 16 server with pgvector 0.8.6)
# All are older than the seven day dependency age policy at the time of pinning (2026-11-05).
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
ver=${PY_VERSION:-3.13}
case "$ver" in
  3.13) default_image=python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 ;; # 3.13.17-slim-trixie, 2026-10-24
  3.12) default_image=python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 ;; # 3.12.15-slim-trixie, 2026-10-22
  *) echo "unsupported PY_VERSION (3.12 or 3.13)" >&2; exit 2 ;;
esac
image=${PY_IMAGE:-$default_image}
pg_image=pgvector/pgvector@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b # 0.8.6-pg16, 2026-08-13
[ $# -ge 1 ] || { echo "usage: py.sh install | py.sh <shell command>" >&2; exit 2; }
mode=ro
[ -z "${PY_RW:-}" ] || mode=rw
name=$(basename "$here")
vol="py-venv-$name-$ver"
mkdir -p "$here/out"
docker volume create "$vol" > /dev/null
# the volume must be owned by the unprivileged user
docker run --rm -u 0 -v "$vol:/venv" "$image" chown "$(id -u):$(id -g)" /venv > /dev/null

if [ "$1" = install ]; then
  net=bridge
  cmd='[ -x /venv/bin/python ] || python -m venv /venv; /venv/bin/python -m pip install --disable-pip-version-check --no-deps --require-hashes -r requirements.txt -r requirements-dev.txt'
else
  net=${PY_NET:-none}
  cmd="$*"
fi

db_env=""
db_password=""
pg=""
cleanup() {
  [ -z "$pg" ] || docker rm -f -v "$pg" > /dev/null 2>&1 || true
  [ -z "${pgnet:-}" ] || docker network rm "$pgnet" > /dev/null 2>&1 || true
}
if [ -n "${PY_DB:-}" ] && [ "$1" != install ]; then
  suffix=$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
  pgnet=rag-chatbot-test-$suffix
  pg=rag-chatbot-test-pg-$suffix
  password=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
  trap cleanup EXIT INT TERM
  docker network create --internal "$pgnet" > /dev/null
  docker run -d --name "$pg" --network "$pgnet" \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 2g --cpus 2 \
    --user 999:999 --read-only \
    --tmpfs /var/lib/postgresql/data:rw,uid=999,gid=999,mode=0700 \
    --tmpfs /var/run/postgresql:rw,uid=999,gid=999,mode=0755 --tmpfs /tmp:rw,mode=1777 \
    -e POSTGRES_PASSWORD="$password" "$pg_image" -c fsync=off -c synchronous_commit=off -c full_page_writes=off > /dev/null
  i=0
  until docker exec "$pg" pg_isready -q -h 127.0.0.1 -U postgres; do
    i=$((i + 1))
    if [ "$i" -gt 120 ]; then echo "py.sh: the database did not become ready" >&2; exit 3; fi
    sleep 0.5
  done
  net="container:$pg"
  db_env="postgresql://postgres@127.0.0.1:5432/postgres"
  db_password=$password
fi

# not `exec`: the cleanup trap has to run after the container
set +e
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus ${PY_CPUS:-4} --memory 6g --pids-limit 1024 \
  --network "$net" \
  --read-only --tmpfs /tmp:rw,exec,size=1g \
  -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e PIP_CACHE_DIR=/venv/.pip-cache -e MYPY_CACHE_DIR=/work/out/.mypy_cache -e RUFF_CACHE_DIR=/work/out/.ruff_cache \
  -e COVERAGE_FILE=/work/out/.coverage -e PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin \
  ${db_env:+-e "RAG_TEST_DATABASE_URL=$db_env" -e "PGPASSWORD=$db_password"} \
  -v "$here:/work:$mode" -v "$here/out:/work/out" -v "$vol:/venv" -w /work \
  "$image" sh -c "$cmd"
status=$?
exit "$status"
