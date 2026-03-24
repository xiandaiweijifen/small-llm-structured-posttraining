# External Component Verifier Batch Summary

- source experiment: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- external train: `D:\project\small-llm-structured-posttraining\data\external_generalization\gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external test: `D:\project\small-llm-structured-posttraining\data\external_generalization\gorkemsevinc_customer_support_tickets_test_reduced.jsonl`

| Preset | Field Exact Match | End-to-End Exact Match | Main Note |
| --- | ---: | ---: | --- |
| guarded_name_majority_p80 | 0.7517 | 0.0636 | high-purity external `name -> component` only |
| component_nb_text_name | 0.7496 | 0.0071 | NB verifier on input text, summary, and name tokens |
| component_nb_text_name_pred | 0.7510 | 0.0489 | NB verifier plus predicted category and priority features |
| hybrid_guarded_or_nb | 0.7510 | 0.0489 | use guarded majority first, then NB fallback |
| hybrid_guarded_vote_nb | 0.7496 | 0.0071 | majority vote over guarded mapping and both NB variants |
