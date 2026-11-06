# Experiment scripts

[galactus-diag.sh](galactus-diag.sh) is the 16-phase diagnostic harness used in the July investigation. Its commands and assumptions belong to that historical configuration; [Session 4](../results/lab-notebook/2026-07-14-1108-v3-diagnostic-run.md) records the run and its failures.

[session-10-rerun.sh](session-10-rerun.sh) records the proposed controlled rerun of the first DSpark sweep. The later measurements in [the August common baseline](../results/lab-notebook/2026-08-15-to-16-common-baseline-2tb.md) superseded that protocol. It remains here as part of the experimental record.

The [reproduction guide](../REPRODUCE.md) covers the model file, build, memory population, effective batch sizes, and placement needed to compare another run with these results.

## October 28 Maya sweeps

[maya-prefill-sweep.py](maya-prefill-sweep.py) is the completed chunk/ring harness. [maya-prefill-tp-sweep.py](maya-prefill-tp-sweep.py) is the subsequent TP/workspace harness, intentionally stopped after 15 completed cases; its partial server results and exact scripts as run are in [the collected archive](../results/raw-logs/2026-10-29-maya-collection/README.md). Both are intended for the Linux Maya host, reserve port 8082 for testing, and retain results under `/opt/maya-test-results`. See [the experiment record](../results/lab-notebook/2026-10-28-maya-prefill-and-regression.md) for conditions, limitations, and the quality constraint.

## November 5 Maya v1.1.54 sweep

[maya_benchmark_1154_fixed.py](maya_benchmark_1154_fixed.py) is the supplied historical runner, retained unchanged. It fetches and checks out v1.1.54 in the supplied Maya repository, refreshes the HIP engine, and runs isolated child configurations; it does not make a separate checkout. Production must already be stopped and remains stopped afterward. The [experiment record](../results/lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md) documents the completed sweep, failed peer retest, dependency and directory-collision failures, and limitations of its blocking reads, cache checks, and quality validation. This notebook update syntax-checks the file but does not execute it or apply the later proposed patches.
