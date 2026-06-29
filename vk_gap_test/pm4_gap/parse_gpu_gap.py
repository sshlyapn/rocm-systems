#!/usr/bin/env python3
# Parse a rocprofv3 --kernel-trace CSV and report GPU-side per-kernel duration
# and inter-kernel gap for the addk chain produced by hip_gpu_gap.
#
# gap[i] = start[i+1] - end[i]  (ns)  -- GPU idle between consecutive dispatches
# dur[i] = end[i]   - start[i]  (ns)  -- kernel execution time on the CP/CU
#
# Timestamps are GPU clock nanoseconds. We keep only the addk kernels, sort by
# start, and summarize. Plain ASCII only.

import csv, sys, statistics

def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    kname = sys.argv[3] if len(sys.argv) > 3 else "addk"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if kname in r["Kernel_Name"]:
                rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    rows.sort()
    if len(rows) < 2:
        print("%s: only %d %s kernels, nothing to measure" % (label, len(rows), kname))
        return
    durs = [(e - s) for (s, e) in rows]
    gaps = [rows[i + 1][0] - rows[i][1] for i in range(len(rows) - 1)]
    gaps = [g for g in gaps if g >= 0]  # drop the rare out-of-order pair
    span = rows[-1][1] - rows[0][0]
    print("%-14s n=%-5d | dur ns: mean=%.0f med=%.0f p99=%.0f | "
          "gap ns: mean=%.0f med=%.0f p99=%.0f min=%.0f | "
          "throughput: %.3f kernels/us"
          % (label, len(rows),
             statistics.mean(durs), statistics.median(durs), pct(durs, 99),
             statistics.mean(gaps), statistics.median(gaps), pct(gaps, 99), min(gaps),
             1000.0 * len(rows) / span))

if __name__ == "__main__":
    main()
