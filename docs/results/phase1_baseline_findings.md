# Phase 1 Baseline Findings

## Full-Schema Prompt-Only

- valid JSON rate: `0.9646`
- schema compliance rate: `0.0000`
- field exact match: `0.2894`
- end-to-end exact match: `0.0000`

Main failure mode:

- mostly missing required schema fields

## Full-Schema Prompt-Only + Repair

- valid JSON rate: `0.9646`
- schema compliance rate: `0.9567`
- field exact match: `0.4685`
- end-to-end exact match: `0.0000`

Interpretation:

- lightweight repair is enough to fix most structural failures
- repair improves schema compliance substantially
- repair does not solve semantic extraction

## Full-Schema QLoRA

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7708`
- end-to-end exact match: `0.0000`

Main failure mode:

- all remaining failures are value hallucination

Interpretation:

- post-training teaches the model to emit schema-compliant JSON reliably
- structure is solved, semantics are not

## Full-Schema QLoRA + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7708`
- end-to-end exact match: `0.0000`

Interpretation:

- repair adds essentially no value after full-schema post-training
- once structure is already correct, repair cannot fix hallucinated content

## Reduced-Schema QLoRA

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8851`
- end-to-end exact match: `0.4882`

By complexity bucket:

- simple exact match: `0.3770`
- medium exact match: `0.5625`
- complex exact match: `0.1176`

Interpretation:

- removing noisy identity fields materially improves learning
- the main bottleneck is not only model size, but also target design and label quality

## Reduced-Schema QLoRA + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8851`
- end-to-end exact match: `0.4882`

Interpretation:

- reduced-schema QLoRA already solves the structure problem
- repair again adds almost no value

## Schema-Conditioned Reduced QLoRA Generalization

- overall valid JSON rate: `0.9980`
- overall schema compliance rate: `0.9980`
- overall field exact match: `0.8764`
- overall end-to-end exact match: `0.4646`

Seen vs unseen schema:

- seen field exact match: `0.8837`
- unseen field exact match: `0.8691`
- seen end-to-end exact match: `0.4764`
- unseen end-to-end exact match: `0.4528`

Interpretation:

- schema-conditioned post-training preserves structure under mild schema shift
- unseen schema causes a small but real semantic drop
- the main generalization bottleneck is still field semantics rather than JSON validity

## Stage 2 Data-Regime Ablation

### Reduced-Schema QLoRA on 600 Training Samples

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7645`
- end-to-end exact match: `0.0394`

Interpretation:

- a small reduced-schema subset is enough to preserve structural learning
- the run collapses mainly on semantic fields rather than formatting
- this is strong evidence that post-training quality depends heavily on data scale and coverage, not only on whether fine-tuning happened at all

Main degraded semantic fields:

- `actions_requested[0].action`: `0.1102`
- `category`: `0.5000`
- `affected_systems[0].component`: `0.5354`
- `priority`: `0.7047`

### Reduced-Schema QLoRA on 600 Training Samples + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7645`
- end-to-end exact match: `0.0394`

Interpretation:

- repair adds nothing once the small-data model already emits structurally valid JSON
- the remaining failure mode is semantic weakness, not residual structure errors

## Stage 2 LoRA Rank Ablation

### Rank 8

- valid JSON rate: `0.9961`
- schema compliance rate: `0.9961`
- field exact match: `0.8604`
- end-to-end exact match: `0.4173`

Interpretation:

- rank 8 underfits the reduced-schema task relative to stronger baselines
- the hardest semantic fields remain noticeably weaker than the best rank 16 and rank 32 runs

### Rank 16

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8844`
- end-to-end exact match: `0.4843`

Interpretation:

- rank 16 is already enough to recover the original reduced-schema baseline level
- this is a good efficiency-quality tradeoff point if the goal is a practical default

### Rank 32

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8912`
- end-to-end exact match: `0.5079`

Interpretation:

- higher LoRA capacity gives a small but real improvement over rank 16
- the gain shows up mostly in the hardest semantic fields and in end-to-end exact match
- rank 32 becomes the strongest non-curriculum Stage 2 run

Rank trend summary:

- rank 8 is clearly worse than rank 16 and rank 32
- rank 16 roughly matches the original reduced-schema baseline
- rank 32 improves semantic-field learning, especially `action`, `category`, and `priority`

## Stage 2 Curriculum Training

### Simple/Medium Then Full Continuation

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9037`
- end-to-end exact match: `0.5315`

Interpretation:

- curriculum training was the strongest run at this stage of the project
- it outperforms the previous H200-fast reduced baseline on end-to-end exact match
- the gain comes from better semantic learning rather than any structural repair effect

Main improved semantic fields:

- `actions_requested[0].action`: `0.6811`
- `affected_systems[0].component`: `0.7953`
- `category`: `0.8268`
- `priority`: `0.8307`

### Curriculum + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9037`
- end-to-end exact match: `0.5315`

Interpretation:

- repair again adds no measurable value once post-training has already solved structure
- the curriculum result strengthens the project's conclusion about training-vs-repair division of labor

## Stage 2 Epoch Ablation

### Rank 16, Epoch 2

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8590`
- end-to-end exact match: `0.3858`

Interpretation:

- structure is already solved by epoch 2
- the remaining weakness is concentrated in semantic fields
- this run is clearly under-trained relative to later epoch settings

### Rank 16, Epoch 5

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9145`
- end-to-end exact match: `0.5709`

Interpretation:

- most of the useful epoch-driven gain is already achieved by epoch 5
- the gain is concentrated in `action`, `component`, `category`, and `priority`
- longer training is improving semantic extraction rather than structural compliance

### Rank 16, Epoch 9

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9166`
- end-to-end exact match: `0.5709`

Interpretation:

- training beyond epoch 5 gives only marginal additional field-level gain
- end-to-end exact match has effectively plateaued by this point
- epoch is important, but the return after epoch 5 is small

## Stage 2 Learning-Rate Ablation

### Rank 16, Epoch 5, LR 1e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8901`
- end-to-end exact match: `0.4882`

Interpretation:

- `1e-4` is too conservative for this setup
- structure is perfect, but semantic convergence is incomplete

### Rank 16, Epoch 5, LR 2e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9141`
- end-to-end exact match: `0.5709`

Interpretation:

- `2e-4` is the strongest balance among the tested single-stage learning-rate settings
- it improves the hardest semantic fields without hurting end-to-end stability

### Rank 16, Epoch 5, LR 4e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9173`
- end-to-end exact match: `0.5591`

Interpretation:

- `4e-4` slightly improves average field accuracy
- but it is less stable on full-sample exact match than `2e-4`
- larger step size helps some fields while hurting exact full-record correctness

## Stage 2 Structure-Then-Semantics Training

### Structure-First, Then Full Semantic Continuation

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9245`
- end-to-end exact match: `0.5787`

Interpretation:

- this is now the strongest pre-canonicalization run in the repository
- explicit staging of structure-focused learning followed by semantic continuation improves the hardest semantic fields further
- the result supports the hypothesis that structure learning and semantic learning benefit from different training phases

Main improved semantic fields:

- `actions_requested[0].action`: `0.7323`
- `affected_systems[0].component`: `0.8701`
- `category`: `0.8780`
- `priority`: `0.8583`

## Stage 3 Hard-Sample Continuation

### Hard Mining Summary

- training records mined: `1993`
- hard records identified: `561`
- hard fraction: `0.2815`

Interpretation:

- the remaining bottleneck really is concentrated in a substantial hard semantic subset
- the strongest staged model still misses key fields on roughly 28 percent of the training set
- this justifies testing targeted continuation, but does not guarantee that naive hard-subset continuation will help

### Hard-Only Continuation

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8726`
- end-to-end exact match: `0.3307`

Interpretation:

- hard-only continuation badly harms overall quality
- removing the full-data distribution is clearly too aggressive

### Full Plus Hard Mix, x2, Epoch 2, LR 1e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9155`
- end-to-end exact match: `0.5433`

Interpretation:

- this is the best Stage 3 continuation result
- it remains below the strongest staged baseline
- lightly mixing hard samples is safer than hard-only continuation, but it still does not improve the final best result

### Full Plus Hard Mix, x3, Epoch 2, LR 1e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9023`
- end-to-end exact match: `0.5039`

Interpretation:

- heavier hard oversampling clearly degrades quality
- pushing the hard subset too hard creates distribution drift and hurts exact full-record correctness

### Full Plus Hard Mix, x2, Epoch 2, LR 5e-5

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9109`
- end-to-end exact match: `0.5354`

Interpretation:

- lowering the learning rate does not recover the best staged baseline
- the main issue here is not just optimization step size; it is the continuation data recipe itself

## Stage 6 Action Canonicalization

### Canonical Action, Single-Stage, Epoch 7, LR 2e-4

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9341`
- end-to-end exact match: `0.6654`

Interpretation:

- this was the first run to become the strongest result in the repository after target canonicalization
- the main gain comes from target redesign, not from repair or hard-example continuation
- canonicalizing `actions_requested[0].action` sharply reduces target entropy and unlocks a large exact-match gain
- under the canonicalized target, a simpler single-stage run is slightly stronger than the staged alternatives

Main improved semantic fields:

- `actions_requested[0].action`: `0.8622`
- `affected_systems[0].component`: `0.8583`
- `category`: `0.8661`
- `priority`: `0.8543`

### Canonical Action, Structure Then Semantics, Stage 2 Epoch 9

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9320`
- end-to-end exact match: `0.6654`

Interpretation:

- staged training remains strong under the canonicalized target
- however, it no longer clearly outperforms the simpler best single-stage setup
- once the hardest `action` target is normalized, training complexity matters less than it did before

## Stage 7 Component Canonicalization Follow-Up

### Canonical Action + Component, Structure Then Semantics, Stage 2 Epoch 9

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9402`
- end-to-end exact match: `0.6772`

Interpretation:

- this is now the strongest run in the repository
- `component` canonicalization is not strong enough on its own, but it becomes useful when combined with the already-validated `action` canonicalization and staged training
- the main gain comes from a large improvement on `affected_systems[0].component`, which is enough to offset a small drop in `action`
- this result strengthens the overall story that target redesign works best when it is both precise and paired with the right training strategy

Main improved semantic fields:

- `affected_systems[0].component`: `0.9173`
- `affected_systems[0].name`: `0.9252`
- `priority`: `0.8701`
- `actions_requested[0].action`: `0.8504`

## Stage 8 Deterministic Postprocessing

### Action + Component Majority Postprocess

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9427`
- end-to-end exact match: `0.6929`

Interpretation:

- this is now the strongest overall result in the repository
- it does not require retraining; it is a deterministic postprocessing pass applied on top of the Stage 7 best predictions
- the main useful rule is `component <- predicted name` using the train-set majority mapping
- refreshing canonical `action` from `category + summary` helps field exact match slightly, but the end-to-end gain is driven mainly by `component` consistency
- this is a useful final result because it cleanly separates training-time gains from a cheap final consistency layer

Main improved semantic fields:

- `affected_systems[0].component`: `0.9370`
- `actions_requested[0].action`: `0.8583`
- overall end-to-end exact match: `0.6929`

## Stage 9 Lexical Postprocessing

### Combined Lexical Postprocess

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9470`
- end-to-end exact match: `0.7205`

Interpretation:

- this is now the strongest overall result in the repository
- it does not require retraining; it is a small lexical rule layer applied on top of the Stage 8 predictions
- the useful gain comes mainly from promoting a small set of clearly severe cases to `priority=urgent` and `constraints.blocking=true`
- lexical `incident` relabeling is not helpful by itself; the Stage 9 gain is primarily a severity gain rather than a category gain

Main improved semantic fields:

- `priority`: `0.9016`
- `constraints.blocking`: `0.9724`
- overall end-to-end exact match: `0.7205`

## Updated Project-Level Conclusion

The combined experiments now support a more refined story:

- prompt-only is weak on schema compliance
- repair is strong for structural normalization
- post-training is the main lever for stable structured generation
- reduced-schema target design was the first major semantic improvement
- staged structure-then-semantics training became the strongest pre-canonicalization baseline
- broad hard-example continuation did not beat that baseline
- canonicalizing the hardest semantic field, `actions_requested[0].action`, is the first change that clearly breaks through the previous end-to-end ceiling
- `component` canonicalization alone is weak, but joint `action + component` canonicalization plus staged training pushes the best result further
- a final deterministic consistency pass on top of the strongest trained run pushes the best result further again, without any new optimization steps
- a final lexical severity-focused postprocess pass pushes the best result further again, again without retraining
- after structure is solved, the strongest remaining levers are target design and semantic-label consistency rather than repair
