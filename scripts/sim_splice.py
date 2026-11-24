#!/usr/bin/env python
"""Splice the generated sim-to-real section and its summary paragraph into GAPS.md,
idempotently.

Run AFTER the sim reports exist (development + held-out). It calls scripts/sim_section.py,
which reads artifacts/rl_acquire/sim/summary*.json and prints markdown, and replaces
whatever sits between the `sim-generated` / `sim-summary` markers -- re-running replaces
the previous splice instead of stacking copies, and no number in either block is typed.

First run: the section goes before "## Case studies" (after the combiner section, whose
cross-line finding it follows from), the summary paragraph after the combiner summary
paragraph. Both anchors must be found exactly, or nothing is written.

Refuses to splice if a held-out summary is missing: a fresh report can be invisible on the
login node for up to 10 minutes (NFS acregmin=600 on /zfsauton/scratch), and the verdicts
would silently be computed on fewer splits.

It edits a text file and does no computation, so it runs on the login node like a git
command. Usage: python3 scripts/sim_splice.py
"""
import os, subprocess, sys

HOME = os.environ.get("PJ_HOME", "/zfsauton2/home/yixiz/10703_pj")
ART = os.environ.get("PJ_ARTIFACTS", "/zfsauton/scratch/yixiz/10703_pj/artifacts")
PY = os.environ.get("PJ_PY_ANALYSIS", "/zfsauton/scratch/yixiz/envs/cellflow/bin/python")
NEED = ("summary.json", "summary_confirm.json", "summary_confirm2.json", "summary_confirm3.json")
B1, E1 = "<!-- sim-generated: begin -->", "<!-- sim-generated: end -->"
B2, E2 = "<!-- sim-summary: begin -->", "<!-- sim-summary: end -->"
SEC_ANCHOR = "## Case studies: the gap and the improvement, gene by gene"
SUM_ANCHOR = "\nThe reward oracle, not the generator, is what limits this project."

missing = [n for n in NEED if not os.path.exists(os.path.join(ART, "rl_acquire", "sim", n))]
if missing:
    sys.exit(f"missing {missing} -- not run yet, or hidden by the 10-minute NFS cache; do not splice")


def gen(part):
    out = subprocess.run([PY, "-I", f"{HOME}/scripts/sim_section.py", "--art-dir", ART, "--part", part],
                         capture_output=True, text=True, check=True).stdout.rstrip()
    if not out.strip():
        sys.exit(f"generator produced nothing for {part}")
    return out


sec, summ = gen("section"), gen("summary")
p = os.path.join(HOME, "GAPS.md")
s = open(p).read()


def put(s, begin, end, body, anchor, before_text):
    block = f"{begin}\n{body}\n{end}"
    if begin in s and end in s:
        a, b = s.index(begin), s.index(end) + len(end)
        print(f"replaced previous {begin}")
        return s[:a] + block + s[b:]
    if s.count(anchor) != 1:
        sys.exit(f"anchor not found exactly once: {anchor!r}; refusing to guess")
    i = s.index(anchor)
    return s[:i] + before_text(block) + s[i:]


s = put(s, B1, E1, sec, SEC_ANCHOR, lambda blk: f"{blk}\n\n---\n\n")
s = put(s, B2, E2, summ, SUM_ANCHOR, lambda blk: f"\n{blk}\n")
open(p, "w").write(s)
print(f"GAPS.md now {len(s)} chars")
