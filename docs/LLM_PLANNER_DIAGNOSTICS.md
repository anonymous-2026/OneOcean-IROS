# LLM Planner Diagnostics

Source run: `V67LLMCOST_planningSuite_medium_cg2_s0-2_20260305_203000`. This is an aggregation of existing artifacts; no new model run was performed.

Recorded protocol: dynamics `6dof`, current gain `2.0`, planner stride `200` steps, and output cap `96` tokens. Each method/task group contains three medium-difficulty seeds.

Heuristic and behavior-cloning methods generate low-level actions. The LLM methods only propose discrete source or waypoint assignments and share deterministic low-level goal following, clipping, dynamics, current, and constraints. Their rows are therefore planning diagnostics, not an architecture-matched end-to-end ranking.

## Task-level results

| Method | Task | Episodes | SR | Time (s) | Energy | Collision | Valid plan | Fallback | Latency/call (ms) | Tokens/call (in/out) | Completion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `heuristic` | `area_scan_terrain_recon` | 3 | 0.333 | 792.0 | 15937.9 | 0.000 | -- | -- | -- | --/-- | 0.747 |
| `heuristic` | `pipeline_inspection_leak_detection` | 3 | 0.667 | 580.0 | 10214.4 | 0.508 | -- | -- | -- | --/-- | 0.667 |
| `heuristic` | `surface_pollution_cleanup_multiagent` | 3 | 0.667 | 581.0 | 9025.4 | 0.474 | -- | -- | -- | --/-- | 0.944 |
| `llm_chatglm3_6b` | `area_scan_terrain_recon` | 3 | 0.667 | 720.0 | 12377.9 | 0.000 | 0.056 | 0.944 | 1718.3 | 610.8/91.7 | 0.773 |
| `llm_chatglm3_6b` | `pipeline_inspection_leak_detection` | 3 | 0.333 | 654.0 | 12494.1 | 0.111 | 0.500 | 0.500 | 1458.1 | 637.6/77.2 | 0.556 |
| `llm_chatglm3_6b` | `surface_pollution_cleanup_multiagent` | 3 | 0.667 | 583.5 | 9048.8 | 0.474 | 0.636 | 0.364 | 1068.1 | 639.3/51.2 | 0.944 |
| `llm_llama2_7b` | `area_scan_terrain_recon` | 3 | 0.000 | -- | 19137.8 | 0.000 | 1.000 | 0.000 | 7400.4 | 620.8/715.8 | 0.707 |
| `llm_llama2_7b` | `pipeline_inspection_leak_detection` | 3 | 0.333 | 427.0 | 11718.0 | 0.002 | 1.000 | 0.000 | 7581.1 | 645.3/740.3 | 0.556 |
| `llm_llama2_7b` | `surface_pollution_cleanup_multiagent` | 3 | 0.667 | 582.5 | 9040.3 | 0.473 | 0.727 | 0.273 | 7247.6 | 649.1/744.1 | 0.944 |
| `llm_llama3_8b` | `area_scan_terrain_recon` | 3 | 0.000 | -- | 19726.0 | 0.007 | 1.000 | 0.000 | 1974.1 | 435.0/456.0 | 0.747 |
| `llm_llama3_8b` | `pipeline_inspection_leak_detection` | 3 | 0.333 | 543.0 | 12181.0 | 0.007 | 1.000 | 0.000 | 2029.8 | 466.0/487.0 | 0.611 |
| `llm_llama3_8b` | `surface_pollution_cleanup_multiagent` | 3 | 1.000 | 666.0 | 9585.0 | 0.014 | 0.333 | 0.667 | 2363.8 | 483.0/508.0 | 1.000 |
| `llm_mistral7b` | `area_scan_terrain_recon` | 3 | 0.000 | -- | 20237.0 | 0.005 | 1.000 | 0.000 | 6105.1 | 605.4/677.7 | 0.747 |
| `llm_mistral7b` | `pipeline_inspection_leak_detection` | 3 | 0.667 | 403.0 | 8714.9 | 0.002 | 0.846 | 0.154 | 4559.4 | 631.0/681.7 | 0.667 |
| `llm_mistral7b` | `surface_pollution_cleanup_multiagent` | 3 | 1.000 | 625.0 | 8993.9 | 0.257 | 0.250 | 0.750 | 3404.9 | 628.8/668.8 | 1.000 |
| `llm_qwen2_7b` | `area_scan_terrain_recon` | 3 | 0.667 | 299.5 | 9851.5 | 0.003 | 0.929 | 0.071 | 2301.4 | 585.6/610.9 | 0.813 |
| `llm_qwen2_7b` | `pipeline_inspection_leak_detection` | 3 | 0.333 | 520.0 | 12154.5 | 0.049 | 1.000 | 0.000 | 2264.3 | 609.8/633.2 | 0.611 |
| `llm_qwen2_7b` | `surface_pollution_cleanup_multiagent` | 3 | 0.667 | 507.5 | 8319.4 | 0.492 | 0.300 | 0.700 | 2669.7 | 615.8/644.8 | 0.944 |
| `llm_qwen2p5_7b` | `area_scan_terrain_recon` | 3 | 0.333 | 1794.0 | 19819.3 | 0.018 | 1.000 | 0.000 | 2112.9 | 587.8/610.5 | 0.760 |
| `llm_qwen2p5_7b` | `pipeline_inspection_leak_detection` | 3 | 0.667 | 879.0 | 11606.1 | 0.004 | 1.000 | 0.000 | 2129.7 | 610.7/632.8 | 0.667 |
| `llm_qwen2p5_7b` | `surface_pollution_cleanup_multiagent` | 3 | 1.000 | 670.0 | 9641.9 | 0.014 | 0.333 | 0.667 | 2439.6 | 616.5/642.7 | 1.000 |
| `mlp_bc` | `area_scan_terrain_recon` | 3 | 0.000 | -- | 10404.4 | 0.000 | -- | -- | -- | --/-- | 0.733 |
| `mlp_bc` | `pipeline_inspection_leak_detection` | 3 | 0.667 | 784.5 | 9061.1 | 0.531 | -- | -- | -- | --/-- | 0.667 |
| `mlp_bc` | `surface_pollution_cleanup_multiagent` | 3 | 0.000 | -- | 3751.7 | 0.281 | -- | -- | -- | --/-- | 0.278 |

## Cross-model task diagnostics

| Task | LLM rows | Mean SR | Mean valid ratio | Mean fallback ratio | Mean latency/call (ms) | Mean completion |
|---|---:|---:|---:|---:|---:|---:|
| `area_scan_terrain_recon` | 6 | 0.278 | 0.831 | 0.169 | 3602.0 | 0.758 |
| `pipeline_inspection_leak_detection` | 6 | 0.444 | 0.891 | 0.109 | 3337.1 | 0.611 |
| `surface_pollution_cleanup_multiagent` | 6 | 0.833 | 0.430 | 0.570 | 3199.0 | 0.972 |

## Interpretation

- `valid_plan_ratio` is the fraction of planner calls that produced a schema-valid assignment; `fallback_ratio` is its complement. A fallback uses the deterministic task allocator, so an LLM-labeled episode is not necessarily controlled by valid LLM plans at every planning event.
- Cleanup tests discrete source assignment and dwell completion. Duplicate or invalid assignments can concentrate agents, which is visible in collision rate; this is not a continuous pollutant-mass experiment.
- Area scan converts discrete waypoint assignments into a continuous coverage objective. A valid assignment can still repeatedly target already covered regions, so schema validity alone does not guarantee coverage efficiency.
- Pipeline inspection requires spatial leak encounters, not only waypoint progress. A planner can produce valid low-frequency assignments while missing leaks between assigned waypoints; completion and waypoint error must be read together.
- Energy is a trajectory-execution proxy, not model inference energy. Uncached latency and token counts are the relevant inference-cost measurements.
- Existing records contain calls, valid/fallback outcomes, latency, and token counts for all three planning-sensitive tasks. The predeclared trigger for a new Qwen3-8B diagnostic is therefore not met, so expanding the model search would add cost without resolving a missing measurement.
