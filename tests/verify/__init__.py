"""
tests.verify
============

The per-milestone verification scripts — one per milestone, each a standalone DEMONSTRATION
rather than an assertion suite: it stages a world, runs it, and prints the A/B evidence the
milestone write-up in `docs/FINDINGS.md` cites.

Run one directly (`AICIV_PROVIDER=random python3 tests/verify/verify_m51.py`) or as a module
(`python3 -m tests.verify.verify_m51`); each puts the repo root on `sys.path` itself, so either
works from anywhere. They are NOT part of `test_simulation.py` and nothing runs them for you.
"""
