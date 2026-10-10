#!/usr/bin/env bash
# Mirror the pod's data into this folder: results/, data/, wandb/ (local W&B run files).
# Incremental: only regular files modified since the last sync are transferred (mtime on the pod).
# Symlinks are skipped: W&B's run folders link to absolute pod paths (results files we already
# copy, /root/.cache logs) that cannot exist here. Model weights (/workspace/hf_cache) and the
# vLLM compile cache are not copied: both are re-downloadable / rebuildable.
# Files deleted on the pod are not deleted here.
#
#   bash runpod/sync_from_pod.sh            # pull
#   bash runpod/sync_from_pod.sh --verify   # pull, then compare file counts and bytes
set -euo pipefail
HOST=runpod
LOCAL=$(cd "$(dirname "$0")/.." && pwd)
STAMP="$LOCAL/.last_pod_sync"
DIRS="results data wandb"

NOW=$(ssh $HOST date +%s)
SINCE=$( [ -f "$STAMP" ] && cat "$STAMP" || echo 0 )
ssh $HOST "cd /workspace/IITJ && find $DIRS -type f -newermt @$SINCE -print0 2>/dev/null | tar cf - --null -T -" \
  | tar xf - -C "$LOCAL"
echo $((NOW - 120)) > "$STAMP"   # 2 min overlap: files being written during the sync are re-sent

if [ "${1:-}" = "--verify" ]; then
  ok=1
  for d in $DIRS; do
    remote=$(ssh $HOST "cd /workspace/IITJ && find $d -type f -printf '%s\n' | awk '{n++; s+=\$1} END {print n+0, s+0}'")
    local_=$(cd "$LOCAL" && find $d -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n+0, s+0}')
    [ "$remote" = "$local_" ] && mark=OK || { mark=DIFF; ok=0; }
    printf "%-8s pod: %-22s local: %-22s %s\n" "$d" "$remote files/bytes" "$local_" "$mark"
  done
  [ $ok = 1 ] && echo "in sync" || echo "differences above (a file may have been written during the sync; re-run)"
fi
