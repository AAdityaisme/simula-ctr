# Rank latency

In-process `rank()` on a warm bundle and character table, one request at a time, no network, no serialization, single core, after one warm-up call. Machine: arm64, python 3.12.13. This is the model-plus-policy cost inside a 50 ms budget, not an end-to-end p99.

| request | candidates | rounds | p50_ms | p95_ms | p99_ms |
| --- | --- | --- | --- | --- | --- |
| shared_creative_two_slots | 3 | 300 | 15.892187 | 17.314390 | 18.013306 |
| fixed_slot_with_gate | 4 | 300 | 15.914229 | 17.185468 | 18.150541 |
| all_unseen_identical | 3 | 300 | 16.256417 | 17.667186 | 18.345735 |
| no_fill | 2 | 300 | 0.239166 | 0.289217 | 0.351884 |
| synthetic_1_candidates | 1 | 300 | 15.221876 | 16.478469 | 17.376835 |
| synthetic_10_candidates | 10 | 300 | 16.199271 | 17.532573 | 18.763853 |
| synthetic_50_candidates | 50 | 300 | 17.756250 | 19.490208 | 20.021482 |
