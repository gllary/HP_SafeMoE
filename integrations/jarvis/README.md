# HP-SafeMoE JARVIS integration

This package implements the five-task JARVIS deployment-aligned validation protocol for HP-SafeMoE.

Training OOF and official-validation evidence calibrate the policy. The locked policy generates the test-role predictions, and `policy/prediction_barrier.json` records their digests before scoring.

Install and run the protocol tests with:

```bash
python -m pip install -e .
python -m pytest -q
```

Official JARVIS data and task definitions are available at <https://jarvis.nist.gov/> and <https://pages.nist.gov/jarvis_leaderboard/>.

The row-level outputs, task metrics, protocol certificate, and independent metric recomputation are provided under `results/JARVIS/` at the repository root.
