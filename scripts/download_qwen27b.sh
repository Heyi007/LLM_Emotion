#!/usr/bin/env bash
# 通过字节内部镜像 huggingface-mirror.bytedance.net 下载完整 Qwen/Qwen3.8-27B 权重。
#
# 镜像不是标准 HF 透明代理：元数据走 /api/models，文件由 TOS bucket 提供，单文件直链：
#   https://huggingface-mirror.bytedance.net/api/bucket/models/Qwen3.8-27B/<filename>
#
# 实测网关会切断长连接（curl 18: transfer closed），单次拉不完 ~4GB 分片；
# 但 <=200MB 的 Range 请求稳定返回 206。因此本脚本采用「分块 Range + 独立重试 + 顺序追加」，
# 每个 HTTP 请求都很小、可重试，从根本上绕开网关的连接时长/字节上限。
#
# 可重复执行：已完整文件跳过；未完成的 .part 按当前字节数续传，已下载内容不浪费。
set -euo pipefail

MIRROR="${MIRROR:-https://huggingface-mirror.bytedance.net}"
BUCKET_BASE="${MIRROR}/api/bucket/models/Qwen3.8-27B"
DEST="${DEST:-/tmp/hf_models/Qwen3.8-27B}"
PARALLEL="${PARALLEL:-3}"          # 同时下载的文件数，降低网关压力
CHUNK="${CHUNK:-209715200}"        # 200 MiB 每块
CHUNK_RETRY="${CHUNK_RETRY:-10}"   # 单块最大重试次数
LOG_DIR="${LOG_DIR:-/tmp/hf_models}"
mkdir -p "$DEST"

LOG="$LOG_DIR/download_$(date +%Y%m%d_%H%M%S).log"
echo "[download] mirror=$MIRROR" | tee "$LOG"
echo "[download] dest=$DEST parallel=$PARALLEL chunk=$((CHUNK/1024/1024))MiB" | tee -a "$LOG"

# 1) 动态拉取 bucket 对象清单。
LIST_JSON="$LOG_DIR/model_files.json"
curl -fsS -m 60 "${MIRROR}/api/bucket/model_files.json" -o "$LIST_JSON"
mapfile -t FILES < <(python3 - "$LIST_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in d.get("Qwen3.8-27B", []):
    name = key.split("/", 2)[-1]
    if name and name != ".metadata.json":
        print(name)
PY
)
echo "[download] ${#FILES[@]} files" | tee -a "$LOG"

# 2) 分块下载单个文件。
download_one() {
    local name="$1"
    local url="${BUCKET_BASE}/${name}"
    local out="${DEST}/${name}"
    local part="${out}.part"
    local chunk="${out}.chunk"
    local expected
    expected="$(curl -fsS -m 30 -I "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')"
    if [[ -s "$out" && "$(stat -c%s "$out")" == "$expected" ]]; then
        echo "SKIP    $name ($expected bytes)"
        return 0
    fi

    local start have end want attempt got
    while true; do
        have="$(stat -c%s "$part" 2>/dev/null || echo 0)"
        if [[ "$have" -ge "$expected" ]]; then break; fi
        start="$have"
        end=$(( start + CHUNK - 1 ))
        [[ "$end" -ge "$expected" ]] && end=$(( expected - 1 ))
        want=$(( end - start + 1 ))
        for attempt in $(seq 1 "$CHUNK_RETRY"); do
            rm -f "$chunk"
            # --retry-all-errors 让 curl 内部也重试；-r 精确请求字节区间
            if curl -fsS --retry-all-errors --retry 3 --retry-delay 1 \
                    --speed-limit 1048576 --speed-time 20 \
                    -r "${start}-${end}" -o "$chunk" "$url"; then
                got="$(stat -c%s "$chunk" 2>/dev/null || echo 0)"
                if [[ "$got" == "$want" ]]; then
                    cat "$chunk" >> "$part"
                    break
                fi
            fi
            echo "  chunk-retry[$attempt] $name @${start} (want $want)" >&2
            sleep 2
            [[ "$attempt" == "$CHUNK_RETRY" ]] && { echo "FAIL    $name at byte $start" >&2; return 1; }
        done
    done

    rm -f "$chunk"
    if [[ "$(stat -c%s "$part")" == "$expected" ]]; then
        mv -f "$part" "$out"
        echo "OK      $name ($expected bytes)"
        return 0
    fi
    echo "FAIL    $name size mismatch" >&2
    return 1
}
export -f download_one
export BUCKET_BASE DEST CHUNK CHUNK_RETRY

# 3) 并行下载（文件级并行，单文件内分块顺序追加）。
if ! printf '%s\n' "${FILES[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'download_one "$@"' _ {} 2>&1 | tee -a "$LOG"; then
    echo "[download] some files failed; re-run this script to resume." | tee -a "$LOG"
    exit 1
fi

# 4) 全量字节数校验。
echo "[download] verifying sizes ..." | tee -a "$LOG"
fail=0
for name in "${FILES[@]}"; do
    expected="$(curl -fsS -m 30 -I "${BUCKET_BASE}/${name}" | grep -i '^content-length' | tail -1 | tr -dc '0-9')"
    actual="$(stat -c%s "${DEST}/${name}" 2>/dev/null || echo 0)"
    if [[ "$actual" != "$expected" ]]; then
        echo "MISMATCH $name expected=$expected actual=$actual" | tee -a "$LOG"
        fail=1
    fi
done
[[ "$fail" -ne 0 ]] && { echo "[download] VERIFICATION FAILED" | tee -a "$LOG"; exit 1; }

du -sh "$DEST" | tee -a "$LOG"
echo "[download] ALL DONE -> $DEST" | tee -a "$LOG"
touch "$DEST/.download_complete"
