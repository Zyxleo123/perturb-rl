#!/usr/bin/env python
"""Splice the generated combiner + curve sub-sections into GAPS.md, idempotently.

Run AFTER `./submit.sh repr` completes. It calls scripts/repr_section.py, which reads
artifacts/repr_*/ and prints markdown, and replaces whatever sits between the
`repr-generated` markers in GAPS.md -- so re-running replaces the previous splice rather
than stacking copies, and no number in that block is ever typed by hand.

It edits a text file and does no computation, so it runs on the login node like a git
command. Paths default to this project's layout; override with PJ_HOME / PJ_ARTIFACTS /
PJ_PY_ANALYSIS.

Usage: python3 scripts/repr_splice.py
"""
import os, subprocess, sys

HOME = os.environ.get("PJ_HOME", "/zfsauton2/home/yixiz/10703_pj")
ART = os.environ.get("PJ_ARTIFACTS", "/zfsauton/scratch/yixiz/10703_pj/artifacts")
PY = os.environ.get("PJ_PY_ANALYSIS", "/zfsauton/scratch/yixiz/envs/cellflow/bin/python")
BEGIN = "<!-- repr-generated: begin -->"
END = "<!-- repr-generated: end -->"
OLD_REPRO = ("Reproduce with `./submit.sh reprprobe` and `./submit.sh reprpred` "
             "(both CPU, ~2 min each) →\n`artifacts/repr_probe/` and `artifacts/repr_predict/`.")
NEW_REPRO = ('Reproduce the whole chain with `./submit.sh repr` (probe → pred → sel → {curve, calib,\n'
             'gram} → figs, all CPU, ~1.5 h) → `artifacts/repr_{probe,predict,select,curve,gram}/` and\n'
             "fig22, then `python3 scripts/repr_splice.py` regenerates this section's generated block.\n"
             'Individual stages: `reprprobe`, `reprpred`, `reprsel`, `reprcurve`, `reprcalib`,\n'
             '`reprgram`, `reprfigs`; `recon` re-checks a per-gene reconciliation failure.')

sec = subprocess.run([PY, "-I", f"{HOME}/scripts/repr_section.py", "--art-dir", ART],
                     capture_output=True, text=True, check=True).stdout.rstrip()
if not sec.strip():
    sys.exit("generator produced nothing")

# Refuse to splice a write-up that is missing the calibration number. Measured
# 2026-11-23: /zfsauton/scratch is NFS v4 mounted with acregmin=acdirmin=600, so a file a
# compute node just wrote can be invisible on this login node for up to TEN MINUTES -- to
# stat, to listdir, and to the generator's subprocess alike, so no in-process re-check can
# see past it. The generator then takes its fallback branch ("measures how much"), which
# omits the number rather than misstating it, and a splice would silently write that.
# Calibration is part of `./submit.sh repr`, so a final write-up must always carry it:
# treat the fallback as a stale view and stop.
if "measures how" in sec:
    sys.exit("the calibration number is missing from the generated section. Either "
             "`./submit.sh reprcalib` has not run, or it ran within the last ~10 minutes and "
             "NFS attribute caching (acregmin=600 on /zfsauton/scratch) still hides "
             "repr_curve/calibration.json from this node. Wait and re-run; do not splice.")

p = os.path.join(HOME, "GAPS.md")
s = open(p).read()

# drop a previous splice if present
if BEGIN in s and END in s:
    a, b = s.index(BEGIN), s.index(END) + len(END)
    s = s[:a] + s[b:]
    s = s.replace("\n\n\n\n", "\n\n")
    print("replaced previous splice")

block = f"{BEGIN}\n{sec}\n{END}\n\n"
if OLD_REPRO in s:
    s = s.replace(OLD_REPRO, block + NEW_REPRO, 1)
    print("spliced before the reproduce line; reproduce line updated")
elif NEW_REPRO in s:
    s = s.replace(NEW_REPRO, block + NEW_REPRO, 1)
    print("spliced before the (already updated) reproduce line")
else:
    sys.exit("could not find the anchor in GAPS.md; refusing to guess")

open(p, "w").write(s)
print(f"GAPS.md now {len(s)} chars")
