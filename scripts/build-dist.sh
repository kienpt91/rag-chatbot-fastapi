#!/bin/sh
# Build the sdist and the wheel in the pinned container, reproducibly, into out/dist.
# usage: scripts/build-dist.sh
#
# The sources are copied to a scratch directory inside the container (the project is mounted read-only) and built
# with `python -m build --no-isolation`: the build backend is the exact, hash-checked setuptools of requirements-dev.txt,
# nothing is downloaded at build time. SOURCE_DATE_EPOCH is the time of the last commit and every copied file gets that
# modification time, the sdist is rewritten with sorted members, a fixed time, no owners and no gzip timestamp: the same commit gives the same bytes.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
epoch=$(git -C "$here" log -1 --format=%ct 2> /dev/null || echo 0)
rm -rf "$here/out/dist"
sh "$here/scripts/py.sh" "set -e
mkdir /tmp/build && cp -r pyproject.toml README.md LICENSE src /tmp/build/ && cd /tmp/build
find . -exec touch -h -d @$epoch {} +
export SOURCE_DATE_EPOCH=$epoch
python -m build --no-isolation --outdir /work/out/dist .
cd /work/out/dist
python - <<'PY'
import glob, gzip, io, os, tarfile
epoch = int(os.environ['SOURCE_DATE_EPOCH'])
for name in glob.glob('*.tar.gz'):
    with tarfile.open(name, 'r:gz') as source:
        members = sorted(source.getmembers(), key=lambda m: m.name)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as target:
            for member in members:
                member.mtime, member.uid, member.gid, member.uname, member.gname = epoch, 0, 0, '', ''
                member.pax_headers = {}
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
    with open(name, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as out:
        out.write(buffer.getvalue())
PY
ls -l"
