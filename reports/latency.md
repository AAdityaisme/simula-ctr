# Rank latency

In-process `rank()` on a warm bundle and character table, one request at a time, no network, no serialization, single core. This is the model-plus-policy cost inside a 50 ms budget, not an end-to-end p99.

| request | candidates | rounds | p50_ms | p95_ms | p99_ms |
| --- | --- | --- | --- | --- | --- |
| shared_creative_two_slots | 3 | 300 | 16.031083 | 16.919688 | 17.858344 |
| fixed_slot_with_gate | 4 | 300 | 16.076125 | 17.077831 | 18.122194 |
| all_unseen_identical | 3 | 300 | 16.423542 | 17.296306 | 17.939258 |
| no_fill | 2 | 300 | 0.233167 | 0.283021 | 0.314131 |
