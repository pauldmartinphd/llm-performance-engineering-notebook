#!/usr/bin/env bash
# session-10-rerun.sh — regenerate the Session 10 table under identical conditions
# Two reps per config, fixed order, baseline at start and end as a drift check,
# TTY captured with script(1). --fit left at default everywhere.
set -e
OUT=/root/s10-rerun; mkdir -p $OUT
M=/models/DeepSeek-V4-Flash/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
D=/models/DeepSeek-V4-Flash/DSpark/DeepseekV4-Flash-20260731-DSpark.gguf
P='Explain how ZFS copy-on-write snapshots work, including the uberblock, block pointers, and space maps.'
BASE="-ngl 99 --cpu-moe -fa on -t 64 -c 8192 -b 8192 -ub 8192 -n 256 --temp 0 -st -no-cnv --perf"
SPEC="--spec-type draft-dspark -md $D -ngld 99"

run() {  # run <tag> <extra flags...>
  local tag=$1; shift
  for rep in 1 2; do
    script -q -a "$OUT/$tag.txt" -c "llama-cli -m $M $BASE -p \"$P\" $*"
  done
}

run baseline-pre
run n1        $SPEC --spec-draft-n-max 1
run n2        $SPEC --spec-draft-n-max 2
run n3        $SPEC --spec-draft-n-max 3
run n5        $SPEC --spec-draft-n-max 5
run n2-p03    $SPEC --spec-draft-n-max 2 --spec-draft-p-min 0.3
run n3-p03    $SPEC --spec-draft-n-max 3 --spec-draft-p-min 0.3
run n3-p05    $SPEC --spec-draft-n-max 3 --spec-draft-p-min 0.5
run baseline-post

# extract: strip terminal control codes, pull the numbers
for f in $OUT/*.txt; do
  echo "== $f"
  col -b < "$f" | grep -aE "Generation|Prompt:|acceptance" | tail -4
done
