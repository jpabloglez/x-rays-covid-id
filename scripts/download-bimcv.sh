#!/usr/bin/env bash
#
# Fetch chest radiographs from the BIMCV-COVID19 b2drop shares, one archive at
# a time: download, extract only the x-ray files, delete the archive, move on.
#
# The reason for the one-at-a-time loop is arithmetic. The two shares total
# 745 GB, and roughly half of the positive partition and 86% of the negative
# one is CT volumes (.nii.gz) that this project has no use for. Those bytes
# cannot be skipped at download time -- they sit inside the same tarballs -- but
# they need never touch the disk. Extracting the PNGs and dropping the archive
# before fetching the next one keeps the working set near 12 GB while ~80 GB
# crosses the wire.
#
# Archives are sharded by subject, so any subset is a coherent sample; there is
# no need to take all 145. Each completed part leaves a .done marker, so the
# script is safe to interrupt and re-run -- it picks up where it stopped.
#
# Usage:
#   ./download-bimcv.sh --partition posi --parts 4
#   ./download-bimcv.sh --partition neg  --parts 12
#
set -euo pipefail

WEBDAV="https://b2drop.bsc.es/public.php/webdav"
TOKEN_POSI="BIMCV-COVID19-cIter_1_2"
TOKEN_NEG="BIMCV-COVID19-cIter_1_2-Negative"

PARTITION="posi"
PARTS=4
DEST=""
KEEP_ARCHIVES=0
DRY_RUN=0

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --parts)     PARTS="$2"; shift 2 ;;
        --dest)      DEST="$2"; shift 2 ;;
        --keep-archives) KEEP_ARCHIVES=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$PARTITION" in
    posi) TOKEN="$TOKEN_POSI" ;;
    neg)  TOKEN="$TOKEN_NEG" ;;
    *) echo "--partition must be 'posi' or 'neg', got '$PARTITION'" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$SCRIPT_DIR/../datasets/bimcv-$PARTITION}"
WORK="$DEST/.archives"
mkdir -p "$DEST" "$WORK"

for tool in curl tar python3; do
    command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

# Ask the share what it actually holds rather than generating part names. The
# negative partition starts at 'partab' with gaps in the sequence, so a
# generated aa..zz range silently requests archives that do not exist.
echo "Listing $PARTITION partition..."
LISTING="$WORK/listing.xml"
curl -sS --max-time 180 -X PROPFIND -u "$TOKEN:" -H "Depth: 1" "$WEBDAV/" -o "$LISTING"

mapfile -t AVAILABLE < <(python3 - "$LISTING" <<'PY'
import re, sys, urllib.parse
xml = open(sys.argv[1]).read()
rows = []
for block in re.findall(r'<d:response>(.*?)</d:response>', xml, re.S):
    href = re.search(r'<d:href>(.*?)</d:href>', block).group(1)
    name = urllib.parse.unquote(href.split('/webdav/')[-1])
    size = re.search(r'<d:getcontentlength>(\d+)</d:getcontentlength>', block)
    if re.search(r'subjects_part[a-z]{2}\.tar\.gz$', name):
        rows.append((name, int(size.group(1)) if size else 0))
for name, size in sorted(rows):
    print(f"{name}\t{size}")
PY
)

if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
    echo "No subject archives found. The share may be under maintenance." >&2
    exit 1
fi

echo "${#AVAILABLE[@]} archives available; taking the first $PARTS."
echo

TOTAL_BYTES=0
DONE=0

for entry in "${AVAILABLE[@]:0:$PARTS}"; do
    NAME="${entry%%$'\t'*}"
    SIZE="${entry##*$'\t'}"
    MARKER="$DEST/.done/$NAME"
    ARCHIVE="$WORK/$NAME"

    if [[ -f "$MARKER" ]]; then
        echo "[skip] $NAME already extracted"
        DONE=$((DONE + 1))
        continue
    fi

    printf '[%d/%d] %s (%.2f GB)\n' "$((DONE + 1))" "$PARTS" "$NAME" "$(echo "$SIZE" | awk '{print $1/1e9}')"
    if [[ $DRY_RUN -eq 1 ]]; then
        DONE=$((DONE + 1))
        continue
    fi

    # --continue-at resumes a partial file, and the retry flags matter here:
    # this share has been observed to stall mid-transfer with the socket open
    # and no bytes moving, which a plain curl would wait on indefinitely.
    echo "  downloading..."
    # --no-progress-meter, not --silent: errors and retry notices still need to
    # reach the log. The meter itself emits a line per second, which across
    # sixteen archives buries every message worth reading under megabytes of
    # carriage returns.
    curl -fSL --no-progress-meter --max-time 7200 --connect-timeout 60 \
         --retry 10 --retry-delay 15 --retry-all-errors \
         --speed-limit 1024 --speed-time 120 \
         --continue-at - -u "$TOKEN:" \
         -o "$ARCHIVE" "$WEBDAV/$NAME"

    # Only the radiographs and the per-study metadata. The .nii.gz CT volumes
    # are the bulk of the archive and are never written to disk.
    echo "  extracting radiographs..."
    tar -xzf "$ARCHIVE" -C "$DEST" --wildcards \
        '*.png' '*.tsv' '*.json' 2>/dev/null || {
            echo "  extraction reported errors; leaving $NAME in place for inspection" >&2
            continue
        }

    if [[ $KEEP_ARCHIVES -eq 0 ]]; then
        rm -f "$ARCHIVE"
    fi

    mkdir -p "$DEST/.done"
    touch "$MARKER"
    TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
    DONE=$((DONE + 1))

    PNGS=$(find "$DEST" -name '*.png' | wc -l)
    echo "  done. $PNGS radiographs on disk, $(du -sh "$DEST" 2>/dev/null | cut -f1) used."
    echo
done

# A stray CT volume would be a bug in the wildcard list rather than a disaster,
# but it is worth saying out loud rather than silently occupying 40 GB.
STRAY=$(find "$DEST" -name '*.nii.gz' 2>/dev/null | wc -l)
if [[ "$STRAY" -gt 0 ]]; then
    echo "warning: $STRAY CT volumes were extracted unexpectedly; remove with:" >&2
    echo "  find '$DEST' -name '*.nii.gz' -delete" >&2
fi

echo "Extracted $DONE archive(s) of the $PARTITION partition."
echo "Radiographs: $(find "$DEST" -name '*.png' | wc -l)"
echo "Disk used:   $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "Downloaded:  $(echo "$TOTAL_BYTES" | awk '{printf "%.1f GB", $1/1e9}') this run."
