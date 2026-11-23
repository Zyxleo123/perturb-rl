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
NEW_REPRO = ("Reproduce the whole chain with `./submit.sh repr` (probe → pred → sel → curve →\n"
             "figs, all CPU, ~45 min) → `artifacts/repr_{probe,predict,select,curve}/` and fig22.\n"
             "Individual stages: `reprprobe`, `reprpred`, `reprsel`, `reprcurve`, `reprfigs`.")

sec = subprocess.run([PY, "-I", f"{HOME}/scripts/repr_section.py", "--art-dir", ART],
                     capture_output=True, text=True, check=True).stdout.rstrip()
if not sec.strip():
    sys.exit("generator produced nothing")

# Guard against a stale filesystem view. Artifacts are written on compute nodes onto
# NFS-backed scratch, and a file another host just created can be invisible here for a
# while (negative-lookup caching): the generator then silently takes its fallback branch
# and the splice writes "measures how much" over a number that is already on disk.
# Observed 2026-11-23. Re-check the file with a fresh listing and refuse if they disagree.
cal = os.path.join(ART, "repr_curve", "calibration.json")
os.listdir(os.path.dirname(cal))                    # forces a fresh directory read
if os.path.exists(cal) and "measures how" in sec:
    sys.exit("calibration.json exists but the generator took its fallback branch: "
             "stale filesystem view, re-run in a moment rather than splicing it")

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
