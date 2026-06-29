#!/usr/bin/env bash
# Pin or restore an AMD GPU clock for reproducible benchmarking.
#
# WHY: the GPU autosuspends when idle (power/control=auto, runtime_status=suspended),
# then wakes and ramps clocks unpredictably on each run. That is the source of the
# bimodal timing noise -- NOT boost-vs-base. Pinning keeps the device awake and
# fixes a performance level.
#
# MUST be run as root on the HOST (sysfs is read-only inside the unprivileged
# container).
#
# See 'gpu_pin_freq.sh help' for usage.
#
# WARNING: this is GLOBAL to the card -- it affects every container/user on this
# GPU. profile_standard lowers the clock and will slow others' workloads. Check
# 'amd-smi process' before running.
set -euo pipefail

usage() {
  cat <<EOF
Pin or restore an AMD GPU clock for reproducible benchmarking.

Usage:
  $0 list
  sudo $0 pin   <gpu_id> [level]
  sudo $0 unpin <gpu_id>
  $0 help

Commands:
  list   List amdgpu GPUs with their ids, names, and card devices.
  pin    Keep the GPU awake and fix a performance level.
  unpin  Restore the original power/clock state saved at pin time.
  help   Show this help.

  <gpu_id> : amdgpu index (0, 1, 2, ...), in the order shown by 'list'.
             Override the device directly with: CARD=cardN sudo $0 ...

  level (pin only, default profile_standard):
    profile_standard  fixed mid clock, below boost   <- recommended (stable)
    profile_peak      highest stable clock (no opportunistic boost)
    profile_min_sclk  lowest sclk
    manual            then set pp_dpm_sclk/pp_dpm_mclk by hand (see note below)
    auto              stock dynamic (boosty)

NOTE (manual specific level): after running with 'manual', do e.g.:
  cat /sys/class/drm/card1/device/pp_dpm_sclk   # list levels
  echo 1 > /sys/class/drm/card1/device/pp_dpm_sclk
  echo 1 > /sys/class/drm/card1/device/pp_dpm_mclk

pin/unpin must run as root on the HOST (sysfs is read-only inside the
unprivileged container).

Passwordless sudo: to run pin/unpin without typing a password, install the
script and add a sudoers entry, e.g.:
  sudo install -o root -g root -m 755 $0 /usr/local/sbin/gpu_pin_freq.sh
  echo '%sudo ALL=(root) NOPASSWD: /usr/local/sbin/gpu_pin_freq.sh' \\
    | sudo tee /etc/sudoers.d/gpu_pin_freq
  sudo chmod 440 /etc/sudoers.d/gpu_pin_freq
Then: sudo gpu_pin_freq.sh pin 0    (no password prompt)
EOF
}

# Read a human-readable GPU name for a card device, best-effort.
gpu_name() {
  local dev="$1" name=""
  name="$(cat "$dev/product_name" 2>/dev/null || true)"
  if [ -z "$name" ]; then
    local mp pid
    mp="$(cat "$dev/uevent" 2>/dev/null | sed -n 's/^PCI_ID=//p')"
    pid="${mp:-?}"
    name="PCI ${pid}"
  fi
  echo "$name"
}

# Enumerate amdgpu cards sorted by card number. Prints "<idx> <card> <dev>".
enumerate_gpus() {
  local idx=0 d
  for d in $(ls -d /sys/class/drm/card[0-9]*/device 2>/dev/null | sort -V); do
    [ -f "$d/uevent" ] || continue
    grep -q '^DRIVER=amdgpu$' "$d/uevent" 2>/dev/null || continue
    echo "$idx $(basename "$(dirname "$d")") $d"
    idx=$((idx + 1))
  done
}

list_gpus() {
  local found=0 line idx card dev
  printf '%-4s %-8s %-40s %s\n' "ID" "CARD" "NAME" "DEVICE"
  while read -r idx card dev; do
    [ -n "$idx" ] || continue
    found=1
    printf '%-4s %-8s %-40s %s\n' "$idx" "$card" "$(gpu_name "$dev")" "$dev"
  done < <(enumerate_gpus)
  [ "$found" -eq 1 ] || echo "No amdgpu GPUs found."
}

ACTION="${1:-}"
case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  list) list_gpus; exit 0 ;;
  pin|unpin) ;;
  *) usage >&2; exit 1 ;;
esac

GPU_ID="${2:-}"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root, e.g. sudo $0" >&2
  exit 1
fi

# Resolve the target card. CARD=cardN overrides the gpu_id lookup.
CARD_DEV=""
if [ -n "${CARD:-}" ]; then
  case "$CARD" in
    card[0-9]|card[0-9][0-9]) ;;
    *) echo "ERROR: invalid CARD '$CARD' (expected cardN)" >&2; exit 1 ;;
  esac
  CARD_DEV="/sys/class/drm/${CARD}/device"
else
  case "$GPU_ID" in
    ''|*[!0-9]*) echo "ERROR: invalid gpu_id '$GPU_ID' (expected a number, e.g. 0)" >&2; usage >&2; exit 1 ;;
  esac
  while read -r idx _ dev; do
    [ -n "$idx" ] || continue
    if [ "$idx" -eq "$GPU_ID" ]; then CARD_DEV="$dev"; break; fi
  done < <(enumerate_gpus)
fi
if [ -z "$CARD_DEV" ] || [ ! -d "$CARD_DEV" ]; then
  echo "ERROR: amdgpu gpu_id '$GPU_ID' not found; pass CARD=cardN explicitly." >&2
  exit 1
fi

CARD_NAME="$(basename "$(dirname "$CARD_DEV")")"
STATE_FILE="/var/tmp/gpu_freq_state.${CARD_NAME}"
PC="$CARD_DEV/power/control"
FPL="$CARD_DEV/power_dpm_force_performance_level"
echo "Target card : $CARD_DEV ($CARD_NAME)"

if [ "$ACTION" = "unpin" ]; then
  PC_VAL="auto"; FPL_VAL="auto"
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    . "$STATE_FILE"
    PC_VAL="${PC_ORIG:-auto}"
    FPL_VAL="${FPL_ORIG:-auto}"
    echo "Loaded saved state from $STATE_FILE"
  else
    echo "No saved state file; falling back to stock defaults (auto/auto)."
  fi
  echo "Restoring   : force_perf=$FPL_VAL  power/control=$PC_VAL"

  # Restore performance level first (device is awake from the pin), then power control.
  echo "$FPL_VAL" > "$FPL" 2>/dev/null || echo auto > "$FPL"
  echo "$PC_VAL"  > "$PC" 2>/dev/null || echo auto > "$PC"

  echo "force_performance_level -> $(cat "$FPL" 2>/dev/null || echo n/a)"
  echo "power/control          -> $(cat "$PC" 2>/dev/null || echo n/a)"
  echo "runtime_status         -> $(cat "$CARD_DEV/power/runtime_status" 2>/dev/null || echo n/a)"

  rm -f "$STATE_FILE" && echo "Removed $STATE_FILE"
  echo "Restored."
  exit 0
fi

# ACTION = pin
LEVEL="${3:-profile_standard}"
case "$LEVEL" in
  auto|low|high|manual|profile_standard|profile_peak|profile_min_sclk|profile_min_mclk) ;;
  *) echo "ERROR: invalid level '$LEVEL'" >&2; exit 1 ;;
esac

# Warn about other GPU users.
if command -v amd-smi >/dev/null 2>&1; then
  echo "--- amd-smi process (verify no other users before pinning) ---"
  amd-smi process 2>/dev/null | grep -iE 'GPU:|PID:|NAME:' | head -20 || true
  echo "-------------------------------------------------------------"
fi

# Save the ORIGINAL state once (do not clobber on re-run).
if [ ! -f "$STATE_FILE" ]; then
  orig_pc="$(cat "$PC" 2>/dev/null || echo auto)"
  orig_fpl="$(cat "$FPL" 2>/dev/null || echo auto)"
  [ -n "$orig_fpl" ] || orig_fpl="auto"
  printf 'CARD_DEV=%s\nPC_ORIG=%s\nFPL_ORIG=%s\n' "$CARD_DEV" "$orig_pc" "$orig_fpl" > "$STATE_FILE"
  echo "Saved original state -> $STATE_FILE (power/control=$orig_pc force_perf=$orig_fpl)"
else
  echo "Original state already saved at $STATE_FILE (kept as-is)."
fi

# 1) Keep the GPU awake so clocks are stable and the level file is writable.
echo on > "$PC"
sleep 1
echo "runtime_status -> $(cat "$CARD_DEV/power/runtime_status" 2>/dev/null || echo n/a)"

# 2) Pin the performance level.
echo "$LEVEL" > "$FPL"
sleep 1
echo "force_performance_level -> $(cat "$FPL" 2>/dev/null || echo n/a)"

echo "--- pp_dpm_sclk ---"; cat "$CARD_DEV/pp_dpm_sclk" 2>/dev/null || true
echo "--- pp_dpm_mclk ---"; cat "$CARD_DEV/pp_dpm_mclk" 2>/dev/null || true
echo
echo "Pinned. Restore the original state with: sudo $0 unpin $GPU_ID"
