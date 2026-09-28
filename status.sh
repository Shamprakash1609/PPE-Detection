#!/bin/bash
# Show training progress for the PPE notebooks.
#
#   ./status.sh           one snapshot
#   ./status.sh -w        refresh every 20s until Ctrl-C
#   ./status.sh -w 5      refresh every 5s
#
# Reads runs/<name>/results.csv, which Ultralytics appends to once per epoch,
# so this works whether the notebook is running in Jupyter, VS Code or headless.

cd "$(dirname "$0")" || exit 1

MODELS="yolo26n:100:01_:8:9:2:cum yolo11n:59:02_:8:9:2:cum ssdlite:59:03_:5:6:8:per yolo11s:70:05_:8:9:2:cum"

# Which notebook is genuinely executing? Match the python process only —
# matching any command line mentioning the filename also hits shell wrappers.
running_prefix() {
  ps -eo command 2>/dev/null \
    | grep -E '^/.*python3? .*run_notebook\.py' \
    | grep -oE '0[1235]_' | head -1
}

show() {
  local active; active="$(running_prefix)"

  printf '\n  PPE TRAINING STATUS                         %s\n' "$(date '+%H:%M:%S')"
  printf '  %s\n' "--------------------------------------------------------------------------"
  printf '  %-11s %-9s %-8s %-9s %-10s %s\n' MODEL STATE EPOCHS mAP50 mAP50-95 BEST
  printf '  %s\n' "--------------------------------------------------------------------------"

  local spec run total prefix csv state line
  for spec in $MODELS; do
    run="${spec%%:*}"
    total="$(echo "$spec" | cut -d: -f2)"
    prefix="$(echo "$spec" | cut -d: -f3)"
    c50="$(echo "$spec" | cut -d: -f4)"
    c95="$(echo "$spec" | cut -d: -f5)"
    csv="runs/$run/results.csv"

    state="idle"
    [ "$active" = "$prefix" ] && state="RUNNING"

    if [ ! -f "$csv" ]; then
      printf '  %-11s %-9s %-8s %-9s %-10s %s\n' "$run" "$state" "0/$total" "-" "-" "-"
      continue
    fi

    line="$(awk -F, -v tot="$total" -v a="$c50" -v b95="$c95" 'NR>1 && NF>5 {
        n++; m50=$a; m95=$b95
        if ($b95+0 > b+0) { b=$b95; be=$1 }
      } END {
        printf "%d/%s|%.4f|%.4f|%.4f @ ep%d", n, tot, m50, m95, b, be
      }' "$csv")"

    [ "$state" = "idle" ] && [ "${line%%/*}" -ge "$total" ] 2>/dev/null && state="done"

    printf '  %-11s %-9s %-8s %-9s %-10s %s\n' "$run" "$state" \
      "$(echo "$line" | cut -d'|' -f1)" \
      "$(echo "$line" | cut -d'|' -f2)" \
      "$(echo "$line" | cut -d'|' -f3)" \
      "$(echo "$line" | cut -d'|' -f4)"
  done
  printf '  %s\n' "--------------------------------------------------------------------------"

  if [ -n "$active" ]; then
    local r
    case "$active" in
      01_) r=yolo26n; a=8; b95=9 ;;
      02_) r=yolo11n; a=8; b95=9 ;;
      03_) r=ssdlite; a=5; b95=6 ;;
      05_) r=yolo11s; a=8; b95=9 ;;
    esac
    printf '  last 5 epochs of %s:\n' "$r"
    awk -F, -v a="$a" -v b95="$b95" 'NR>1 && NF>5 {
        printf "    epoch %-4d  mAP50 %.4f   mAP50-95 %.4f\n",$1,$a,$b95
      }' "runs/$r/results.csv" 2>/dev/null | tail -5
    # ETA from mean epoch time. Ultralytics stores cumulative seconds; the
    # SSDLite loop stores this-epoch seconds, hence the two modes.
    for spec in $MODELS; do
      [ "${spec%%:*}" = "$r" ] || continue
      etot="$(echo "$spec" | cut -d: -f2)"
      tcol="$(echo "$spec" | cut -d: -f6)"
      tmode="$(echo "$spec" | cut -d: -f7)"
      awk -F, -v tot="$etot" -v tc="$tcol" -v mode="$tmode" 'NR>1 && NF>5 {
          n++; if (mode=="cum") tot_s=$tc; else tot_s+=$tc
        } END {
          if (n>0 && n<tot)
            printf "    eta ~%.0f min for the remaining %d epochs  (%.0fs/epoch)\n",
                   (tot_s/n)*(tot-n)/60, tot-n, tot_s/n
        }' "runs/$r/results.csv" 2>/dev/null
    done
  else
    printf '  nothing training right now\n'
  fi
  printf '\n'
}

if [ "$1" = "-w" ]; then
  while true; do clear; show; sleep "${2:-20}"; done
else
  show
fi
