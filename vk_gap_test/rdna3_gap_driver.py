#!/usr/bin/env python3
# RDNA3 (gfx1100) dispatch-gap driver: replicates the RDNA4 report methodology.
#
#   busy  = median per-kernel duration from rocprofv3 --kernel-trace (forces AQL).
#   e2e   = profiler-free whole-graph wall (best-of-7 inside the binary), PM4 off/on.
#   gap   = (e2e_us - sum_kernel_us) / (K-1)         per-interval, us.
#   saves = AQL gap - PM4 gap (us); e2e reduction (%).
#
# PM4 is invisible to rocprofv3 (raw IB bypasses the traced AQL queue), so busy is
# always taken from the AQL trace and reused for the PM4 column. Plain ASCII only.

import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import tempfile

OURLIB = "/home/sshliapn/code/rocm-systems-rebase/projects/clr/build-develop/hipamd/lib"
HERE = os.path.dirname(os.path.abspath(__file__))
ENV_BASE = dict(os.environ)
ENV_BASE["HIP_VISIBLE_DEVICES"] = "0"
ENV_BASE["LD_LIBRARY_PATH"] = OURLIB


def run(binary, args, pm4, reps=1):
    """Run a gap binary, return median e2e_us over `reps` invocations."""
    e = dict(ENV_BASE)
    e.pop("HIP_PM4_GRAPH", None)
    if pm4:
        e["HIP_PM4_GRAPH"] = "1"
    vals = []
    for _ in range(reps):
        out = subprocess.run([os.path.join(HERE, binary)] + [str(a) for a in args],
                             env=e, capture_output=True, text=True).stdout
        m = re.search(r"e2e_us=([0-9.]+)", out)
        if m:
            vals.append(float(m.group(1)))
    return statistics.median(vals) if vals else float("nan")


def busy(binary, args, krp):
    """rocprofv3 --kernel-trace under AQL. Returns {label: median_us} grouped by
    normalized kernel name (gap_kernel<N>, k1..k4)."""
    e = dict(ENV_BASE)
    e.pop("HIP_PM4_GRAPH", None)
    d = tempfile.mkdtemp(prefix="rp_")
    cmd = ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", d, "-o", "kt",
           "--", os.path.join(HERE, binary)] + [str(a) for a in args[:-1]] + [str(krp)]
    subprocess.run(cmd, env=e, capture_output=True, text=True)
    files = glob.glob(os.path.join(d, "**", "*kernel_trace*.csv"), recursive=True)
    groups = {}
    if files:
        with open(files[0]) as f:
            for r in csv.DictReader(f):
                name = r["Kernel_Name"]
                dur = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0
                lbl = norm_label(name)
                if lbl:
                    groups.setdefault(lbl, []).append(dur)
    subprocess.run(["rm", "-rf", d])
    return {k: statistics.median(v) for k, v in groups.items()}


def norm_label(name):
    if "gap_kernel" in name:
        m = re.search(r"gap_kernel[<I][Li ]*(\d+)", name)
        return "gap%s" % (m.group(1) if m else "?")
    m = re.search(r"\bk([1-4])\b", name)
    if m:
        return "k%s" % m.group(1)
    return None


def gpk(e2e_us, kernel_sum_us, K):
    return (e2e_us - kernel_sum_us) / (K - 1)


# spin, K grid mirroring the RDNA4 report.
UNIFORM = [(250, 1500), (500, 1500), (750, 1000), (1000, 800), (1500, 800),
           (2000, 600), (3000, 400), (5000, 300), (10000, 150), (25000, 80),
           (50000, 50), (100000, 40), (200000, 30), (210000, 30)]

MIX = [(250, 5000, 400), (500, 20000, 400), (250, 50000, 400), (1000, 100000, 400)]

HETERO = [(250, 800), (500, 800), (1000, 800), (2000, 800)]


def do_uniform(free=False):
    if free:
        print("| spin | K | busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |")
        print("|---|---|---|---|---|---|")
    else:
        print("| spin | K | busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |")
        print("|---|---|---|---|---|---|---|---|")
    for spin, K in UNIFORM:
        krp = min(K, 200)
        b = busy("hip_gap_graph.x", [spin, 16384, K], krp)
        bu = statistics.median(list(b.values())) if b else float("nan")
        reps = 5 if free else 1
        eaql = run("hip_gap_graph.x", [spin, 16384, K], False, reps)
        epm4 = run("hip_gap_graph.x", [spin, 16384, K], True, reps)
        ksum = bu * K
        save_pct = (eaql - epm4) / eaql * 100.0
        if free:
            print("| %d | %d | %.2f | %.2f | %.2f | %.1f%% |" %
                  (spin, K, bu, eaql / 1000.0, epm4 / 1000.0, save_pct))
        else:
            gaql = gpk(eaql, ksum, K)
            gpm4 = gpk(epm4, ksum, K)
            print("| %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f us (%.1f%%) |" %
                  (spin, K, bu, eaql / 1000.0, epm4 / 1000.0, gaql, gpm4,
                   gaql - gpm4, save_pct))
        sys.stdout.flush()


def do_mix(free=False):
    if free:
        print("| short | long | busy_s (us)* | busy_l (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |")
        print("|---|---|---|---|---|---|---|")
    else:
        print("| short | long | busy_s (us) | busy_l (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 saves |")
        print("|---|---|---|---|---|---|---|---|")
    for short, lng, K in MIX:
        krp = min(K, 200)
        b = busy("hip_gap_mix.x", [short, lng, 16384, K], krp)
        # gap1 = short kernel, gap3 = long kernel
        bs = b.get("gap1", float("nan"))
        bl = b.get("gap3", float("nan"))
        reps = 5 if free else 1
        eaql = run("hip_gap_mix.x", [short, lng, 16384, K], False, reps)
        epm4 = run("hip_gap_mix.x", [short, lng, 16384, K], True, reps)
        ksum = (K // 2) * bs + (K // 2) * bl
        save_pct = (eaql - epm4) / eaql * 100.0
        if free:
            print("| %d | %d | %.2f | %.2f | %.2f | %.2f | %.1f%% |" %
                  (short, lng, bs, bl, eaql / 1000.0, epm4 / 1000.0, save_pct))
        else:
            gaql = gpk(eaql, ksum, K)
            gpm4 = gpk(epm4, ksum, K)
            print("| %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f us (%.1f%%) |" %
                  (short, lng, bs, bl, eaql / 1000.0, epm4 / 1000.0, gaql,
                   gaql - gpm4, save_pct))
        sys.stdout.flush()


def do_hetero(free=False):
    if free:
        print("| spin | avg busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |")
        print("|---|---|---|---|---|")
    else:
        print("| spin | avg busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |")
        print("|---|---|---|---|---|---|---|")
    for spin, K in HETERO:
        krp = min(K, 200)
        b = busy("hip_gap_hetero.x", [spin, 16384, K], krp)
        vals = [b.get("k%d" % i, float("nan")) for i in (1, 2, 3, 4)]
        avg = statistics.mean(vals)
        reps = 5 if free else 1
        eaql = run("hip_gap_hetero.x", [spin, 16384, K], False, reps)
        epm4 = run("hip_gap_hetero.x", [spin, 16384, K], True, reps)
        ksum = (K // 4) * sum(vals)
        save_pct = (eaql - epm4) / eaql * 100.0
        if free:
            print("| %d | %.2f | %.2f | %.2f | %.1f%% |" %
                  (spin, avg, eaql / 1000.0, epm4 / 1000.0, save_pct))
        else:
            gaql = gpk(eaql, ksum, K)
            gpm4 = gpk(epm4, ksum, K)
            print("| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f us (%.1f%%) |" %
                  (spin, avg, eaql / 1000.0, epm4 / 1000.0, gaql, gpm4,
                   gaql - gpm4, save_pct))
        sys.stdout.flush()


def do_hetero_busy():
    print("| kernel | busy @ spin=500 | busy @ spin=2000 |")
    print("|---|---|---|")
    b500 = busy("hip_gap_hetero.x", [500, 16384, 800], 200)
    b2000 = busy("hip_gap_hetero.x", [2000, 16384, 800], 200)
    for i in (1, 2, 3, 4):
        print("| k%d | %.2f us | %.2f us |" %
              (i, b500.get("k%d" % i, float("nan")), b2000.get("k%d" % i, float("nan"))))


if __name__ == "__main__":
    mode = sys.argv[1]
    {
        "uniform-locked": lambda: do_uniform(False),
        "uniform-free": lambda: do_uniform(True),
        "mix-locked": lambda: do_mix(False),
        "mix-free": lambda: do_mix(True),
        "hetero-locked": lambda: do_hetero(False),
        "hetero-free": lambda: do_hetero(True),
        "hetero-busy": do_hetero_busy,
    }[mode]()
