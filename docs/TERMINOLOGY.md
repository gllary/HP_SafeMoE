# Terminology

## Model name

**HP-SafeMoE** means **Heterogeneous-Provider Safe Mixture-of-Experts**.

- **Heterogeneous provider**: an independently trained Stage-1 task expert. Providers may use different architectures, inputs, and feature spaces.
- **Stage-1 anchor**: the frozen prediction from the provider assigned to the target task.
- **Cross-expert source**: an available non-target provider used during Stage-2 residual construction.
- **Raw Stage-2 proposal**: the anchor-relative prediction produced by the Stage-2 proposal model before safety control.
- **Stage-2 residual**: the bounded difference between the raw Stage-2 proposal and the Stage-1 anchor.
- **Sample-conditioned branch**: source selection based on the current material's expert tokens, target-task embedding and reliability summaries.
- **Relation-augmented branch**: source selection based on the same sample-specific inputs plus a target--source relation learned from outer-training data.
- **Task safety acceptance**: the OOF certificate that determines whether a task-level correction is allowed.
- **Proposal-head residual**: the branch correction added to the anchor inside each fitted proposal member. Its absolute bound is applied in standardized target space for regression and in logit space for classification.
- **Ensemble raw proposal**: the arithmetic mean of member predictions after each member has been restored to the reported output scale. For classification, this is the mean of member probabilities after sigmoid, not the sigmoid of the mean member logit.
- **OOF residual scale**: the OOF-selected multiplier applied to the residual between the restored anchor and the ensemble raw proposal. Regression scaling is affine-equivalent in restored and standardized spaces; classification scaling is a probability-space interpolation after per-member sigmoid restoration and probability averaging.
- **Benefit gate**: a cross-fitted, sample-level estimate of whether applying the proposed residual is likely to improve the Stage-1 anchor. It is fixed to one for classification tasks.
- **Exact fallback**: a rejected or zero-scaled correction returns exactly the Stage-1 anchor.
- **No-cross-expert control**: a matched Stage-2 model in which non-target expert tokens and the learned task--source relation are disabled. The anchor token, target-private residual heads, router and safeguard remain active. Its eight-dimensional reliability vector still contains coarse pool-level summaries of provider predictions, uncertainties and availability; it therefore removes source-specific token access rather than every aggregate statistic derived from the provider pool.
- **Operational safety**: the locked, cross-fitted residual-scale, task-certificate and benefit-gate rule used in the evaluated setting. It is not a formal or distribution-free guarantee.

Matbench and JARVIS use Top-4 sparse attention.

## Public field names

The checkpoint and array schemas use `data_attention`, `data_context`, `DATA_ROUTE` and `use_data_branch` for the Sample-conditioned branch. They use `relation_attention`, `learned_relation_attention`, `relation_context`, `relation_residual`, `relation_standard`, `RELATION_ROUTE`, and `use_relation_branch` for the Relation-augmented branch. These internal names do not denote different methods.

The JARVIS public row-level schema uses `stage1_anchor_prediction`, `raw_stage2_prediction`, `hpsafemoe_prediction`, `data_attention_source_*`, and `learned_relation_attention_source_*`. Its complete mapping is in `results/JARVIS/field_dictionary.csv`.

In the JARVIS row-level schema, `task_safety_accepted` records task safety acceptance and `correction_accepted` records the sample-level benefit-gate decision.

Inference uses the hard selected route. Manuscript source-weight analyses instead use a soft-routing diagnostic: branch attention is weighted by the corresponding soft route probability and set to zero when the correction is rejected. These diagnostic weights summarize fitted source use, while predictive benefit is assessed separately at pool level with the matched Full versus No-cross-expert comparison and OOF safety evidence.

`descriptor_similarity_analysis` records the descriptor-similarity comparison used in the representation analysis.
