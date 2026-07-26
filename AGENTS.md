# Project Instructions

This checkout runs a bounded three-way paper comparison around FinRL-Trading.

- Preserve comparison integrity before adding strategies or retraining models.
- Keep target and actual weights distinct so missing fills remain visible.
- Compare strategies only on shared chronological dates with current SPY and
  QQQ data.
- Treat defensive cash as cash, not synthetic benchmark exposure.
- Keep the RL path offline unless the user explicitly changes that boundary.
- Treat small observation counts as preliminary evidence.
- Use the WSL environment and `uv` for Python verification. Run targeted tests,
  Ruff, and Ty after relevant changes.

## AI Wiki Research Context

The canonical project bridge is available at either host path:

- Windows/Codex:
  `C:\Users\paxto\ai-wiki\wiki\projects\finrl-trading.md`
- WSL: `/mnt/c/Users/paxto/ai-wiki/wiki/projects/finrl-trading.md`

Consult it before proposing changes to strategy architecture, evaluation
metrics, benchmark handling, RL scope, data sources, or paper execution. Follow
only relevant links from the bridge; do not load the entire wiki for routine
mechanical work.

The repository's current data, weights, and metrics remain authoritative.
Reverify fast-moving wiki claims against current primary sources. Clearly
separate source-backed findings, inference, and recommendations. New model or
strategy popularity does not justify widening the comparison before the current
three have adequate observations.

When a wiki-derived idea materially affects a decision, mention the connection
and offer to record the adopted or rejected outcome in the bridge unless the
current task already authorizes wiki updates.
