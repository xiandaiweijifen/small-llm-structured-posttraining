# Stage 2 Results Review

## Missing Reports

- none

## Leaderboard

| Experiment | Num Samples | Valid JSON | Schema Compliance | Field Exact Match | End-to-End Exact Match |
| --- | --- | --- | --- | --- | --- |
| stage2_curriculum | 254 | 1.0 | 1.0 | 0.9037 | 0.5315 |
| stage2_curriculum_repair | 254 | 1.0 | 1.0 | 0.9037 | 0.5315 |
| stage2_rank32 | 254 | 1.0 | 1.0 | 0.8912 | 0.5079 |
| stage2_rank32_repair | 254 | 1.0 | 1.0 | 0.8912 | 0.5079 |
| qlora_reduced_h200fast | 254 | 1.0 | 1.0 | 0.8919 | 0.4961 |
| qlora_reduced | 254 | 1.0 | 1.0 | 0.8851 | 0.4882 |
| qlora_reduced_repair | 254 | 1.0 | 1.0 | 0.8851 | 0.4882 |
| stage2_rank16 | 254 | 1.0 | 1.0 | 0.8844 | 0.4843 |
| stage2_rank16_repair | 254 | 1.0 | 1.0 | 0.8844 | 0.4843 |
| schema_generalization | 508 | 0.998 | 0.998 | 0.8764 | 0.4646 |
| stage2_rank8 | 254 | 0.9961 | 0.9961 | 0.8604 | 0.4173 |
| stage2_rank8_repair | 254 | 0.9961 | 0.9961 | 0.8604 | 0.4173 |
| stage2_data_small_600 | 254 | 1.0 | 1.0 | 0.7645 | 0.0394 |
| stage2_data_small_600_repair | 254 | 1.0 | 1.0 | 0.7645 | 0.0394 |
| qlora_full | 254 | 1.0 | 1.0 | 0.7708 | 0.0 |
| qlora_full_repair | 254 | 1.0 | 1.0 | 0.7708 | 0.0 |
| prompt_only_repair | 254 | 0.9646 | 0.9567 | 0.4685 | 0.0 |
| prompt_only | 254 | 0.9646 | 0.0 | 0.2894 | 0.0 |

## Repair Deltas

| Experiment | Schema Delta | Field Delta | E2E Delta |
| --- | --- | --- | --- |
| prompt_only | 0.9567 | 0.1791 | 0.0 |
| qlora_full | 0.0 | 0.0 | 0.0 |
| qlora_reduced | 0.0 | 0.0 | 0.0 |
| stage2_data_small_600 | 0.0 | 0.0 | 0.0 |
| stage2_curriculum | 0.0 | 0.0 | 0.0 |
| stage2_rank8 | 0.0 | 0.0 | 0.0 |
| stage2_rank16 | 0.0 | 0.0 | 0.0 |
| stage2_rank32 | 0.0 | 0.0 | 0.0 |

## Worst Fields

### qlora_reduced

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.6142 | 98 | 254 |
| affected_systems[0].component | 0.7598 | 61 | 254 |
| category | 0.7953 | 52 | 254 |
| affected_systems[0].name | 0.8228 | 45 | 254 |
| priority | 0.8228 | 45 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| constraints.environment | 0.9646 | 9 | 254 |
| actions_requested[0].deadline | 1.0 | 0 | 254 |

### qlora_reduced_h200fast

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.6457 | 90 | 254 |
| affected_systems[0].component | 0.7677 | 59 | 254 |
| category | 0.811 | 48 | 254 |
| priority | 0.8346 | 42 | 254 |
| affected_systems[0].name | 0.8425 | 40 | 254 |
| constraints.environment | 0.9528 | 12 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| actions_requested[0].deadline | 1.0 | 0 | 254 |

### schema_generalization

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.624 | 191 | 508 |
| affected_systems[0].component | 0.7185 | 143 | 508 |
| category | 0.7697 | 117 | 508 |
| priority | 0.8071 | 98 | 508 |
| affected_systems[0].name | 0.8327 | 85 | 508 |
| customer_impact | 0.8346 | 42 | 254 |
| constraints.blocking | 0.9567 | 22 | 508 |
| constraints.environment | 0.9626 | 19 | 508 |

### stage2_data_small_600

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.1102 | 226 | 254 |
| category | 0.5 | 127 | 254 |
| affected_systems[0].component | 0.5354 | 118 | 254 |
| affected_systems[0].name | 0.6535 | 88 | 254 |
| priority | 0.7047 | 75 | 254 |
| constraints.environment | 0.9528 | 12 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| summary | 0.9961 | 1 | 254 |

### stage2_curriculum

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.6811 | 81 | 254 |
| affected_systems[0].component | 0.7953 | 52 | 254 |
| category | 0.8268 | 44 | 254 |
| priority | 0.8307 | 43 | 254 |
| affected_systems[0].name | 0.878 | 31 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| constraints.environment | 0.9724 | 7 | 254 |
| actions_requested[0].deadline | 1.0 | 0 | 254 |

### stage2_rank8

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.5591 | 112 | 254 |
| affected_systems[0].component | 0.689 | 79 | 254 |
| affected_systems[0].name | 0.7559 | 62 | 254 |
| category | 0.7638 | 60 | 254 |
| priority | 0.8031 | 50 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| constraints.environment | 0.9606 | 10 | 254 |
| summary | 0.9921 | 2 | 254 |

### stage2_rank16

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.622 | 96 | 254 |
| affected_systems[0].component | 0.7638 | 60 | 254 |
| category | 0.7874 | 54 | 254 |
| affected_systems[0].name | 0.8189 | 46 | 254 |
| priority | 0.8228 | 45 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| constraints.environment | 0.9567 | 11 | 254 |
| actions_requested[0].deadline | 1.0 | 0 | 254 |

### stage2_rank32

| Field | Match Rate | Mismatch Count | Total |
| --- | --- | --- | --- |
| actions_requested[0].action | 0.6378 | 92 | 254 |
| affected_systems[0].component | 0.7717 | 58 | 254 |
| category | 0.7992 | 51 | 254 |
| priority | 0.8386 | 41 | 254 |
| affected_systems[0].name | 0.8465 | 39 | 254 |
| constraints.environment | 0.9528 | 12 | 254 |
| constraints.blocking | 0.9567 | 11 | 254 |
| actions_requested[0].deadline | 1.0 | 0 | 254 |

## Generalization Seen vs Unseen

| Split | Num Samples | Valid JSON | Schema Compliance | Field Exact Match | End-to-End Exact Match |
| --- | --- | --- | --- | --- | --- |
| seen | 254 | 1.0 | 1.0 | 0.8837 | 0.4764 |
| unseen | 254 | 0.9961 | 0.9961 | 0.8691 | 0.4528 |
