# Run the Visual Product Search feasibility pilot

## User objective

Run a feasibility pilot for **Visual Product Search** using the DeepFashion In-shop dataset and frozen pretrained CLIP ViT-B/32.

1. Verify dataset access, annotations, product identities, and query/gallery splits.
2. Run baseline image retrieval using CLIP embeddings and cosine similarity.
3. Evaluate Recall@1 and Recall@5, avoiding query/gallery leakage and misleadingly easy subsets.
4. Visualize at least 20 successful and failed retrieval examples.
5. Analyze the main failure patterns and potential improvement opportunities.

Do not fine-tune, train additional models, or build a web interface yet.

Provide a concise report with actual results, limitations, computational cost, and a **Go / Change Dataset / Change Topic** recommendation.

The goal is to determine whether this topic is worth pursuing for FCVG101, not to complete the coursework.

## 1. Access and compute preflight

- Verify current official release, source URLs and terms. If using a mirror, verify release provenance, identities, annotation agreement, completeness and reuse terms; clearly identify it. Never bypass access controls or accept agreements on the user's behalf.
- Record runtime hardware, Python/package versions, available CPU/GPU, RAM, disk, network restrictions and checkpoint availability. Cloud coding access is not evidence of GPU access.
- Do not book paid compute. If CPU is the only option, measure a small representative inference batch and estimate full-run cost before scaling. If impractical, prepare the same runnable code for the user's Colab/Kaggle environment and report what still needs execution.
- Obtain image files and the official identity/evaluation partition annotation (commonly Eval/list_eval_partition.txt). Use bbox annotations only for auditing or a separately labeled future proposal; baseline uses full images with the model's standard preprocessing.

## 2. Dataset audit

- Parse annotation headers robustly, validate declared versus parsed counts, allowed split values, unique paths and valid identities. Reject path traversal, missing/corrupt images and accidental alternate dataset releases.
- Record image and unique-product counts for train, query and gallery; verify these against the official release rather than copying expected numbers into results.
- Assert query/gallery path sets are disjoint, train identities are disjoint from evaluation identities, and every evaluated query identity has a gallery positive.
- Check cross-split exact file hashes and decoded-pixel hashes. Inspect suspicious near-duplicates with an inexpensive image hash, reporting the threshold and manual review limits. Shared product IDs across query/gallery are intentional positives.
- Preserve official split results as the benchmark. If exact image duplicates occur, also provide a clearly labeled sensitivity evaluation excluding those matches. Report queries losing all positives and their treatment; never silently drop them to improve scores.

## 3. Frozen CLIP baseline

- Use pretrained OpenAI CLIP ViT-B/32 weights (OpenAI CLIP package or an equivalent implementation explicitly loading the OpenAI checkpoint). Do not substitute a LAION or fashion-tuned checkpoint. Record model ID, weight checksum/revision and preprocessing.
- Set eval mode, freeze parameters and use inference/no-grad mode. Encode RGB images using the model's standard resize/crop and normalization. L2-normalize image embeddings; cosine similarity is their dot product.
- Cache embeddings with image-order manifest, annotation checksum, model and preprocessing metadata; invalidate incompatible caches. Use batched exact retrieval rather than an approximate index for this pilot.
- Prefer all official queries against the entire official gallery. If resources force query sampling, choose a deterministic outcome-independent sample spanning identities/categories, keep the entire gallery, publish its manifest/seed/coverage and call the result a partial estimate. Never restrict gallery to the sampled query identities or use only easy categories.
- Record download, embedding and retrieval timings separately, peak RAM/VRAM where measurable, disk/cache size and inference throughput. Distinguish observed measurements from extrapolations.

## 4. Evaluation and correctness

- A hit means an exact product-identity match, not a similar category/color. Recall@K is the fraction of eligible queries with at least one correct product among the top K gallery images. Report K=1 and K=5, denominators and the number of excluded/invalid queries with reasons.
- Keep the full gallery's distractors. Mask prohibited identical-image matches for the separate clean sensitivity evaluation, with explicit handling if a query loses all positives.
- Save per-query top-5 paths, IDs, cosine scores and first-positive rank (or a clear lower bound if only top-5 is computed). Include optional per-category/macro-identity summaries to expose imbalance without replacing headline query-level scores.
- Test metric computation with tiny known embeddings/IDs, including multiple gallery positives, absent positives, self-match masking and top-5 correctness. Test parser/split validation and cache-manifest consistency. These tests are not experiment results.
- Explain that unknown CLIP pretraining overlap cannot be ruled out; avoiding evaluation leakage in this pipeline does not prove the pretrained model never saw these images.

## 5. Examples and error analysis

- Produce at least 20 total inspected query examples spanning successes and failures; aim for 10 top-1 successes and 10 top-1 failures, and 20 of each if practical. If a group has too few examples, show all and report the shortfall.
- For each example show query, top-5 retrievals, correct/incorrect product IDs and cosine scores, plus a known gallery positive when top-5 fails. Distinguish top-1 failure recovered at top-5 from failure at both.
- Select cases deterministically within outcome groups with category/identity coverage; publish selected query paths and selection policy. This balanced visualization is diagnostic, not an estimate of failure prevalence.
- Inspect the actual images. Discuss observed patterns such as near-identical garments, color/texture confusion, pose/occlusion, crop loss, background/model shortcuts and annotation ambiguity only when supported by examples. Mark explanations as hypotheses where causality is untested.
- Propose a few targeted next experiments without running training or expanding scope. State why In-shop results may not transfer to user-phone consumer-to-shop search.

## 6. Deliverables and decision

Provide runnable code and commands, pinned/resolved environment, source/provenance and split audit, metrics JSON, per-query results, example selection manifest, private visual artifact access/regeneration instructions, and a concise Vietnamese report.

The report must separate observed results from unrun work and include:

- Actual Recall@1/5, evaluated query/gallery sizes and duplicate sensitivity results if applicable.
- Hardware, runtime, memory/storage, measured compute cost and any unpriced subscription usage (do not invent a dollar price).
- Representative successes/failures, limitations and realistic learning opportunities for a one-person FCVG101 project.
- **Go** if access, compute, evaluation and useful error analysis support pursuing it; **Change Dataset** if access or task mismatch is the main barrier; **Change Topic** only when the evidence supports abandoning the problem itself. Low baseline recall alone is not a reason to abandon a useful learning topic.

If access or execution remains blocked, state that feasibility is not yet established, avoid invented metrics, give a conditional recommendation and the exact next user action. Push code/report to a reviewable branch; do not call the pilot complete until its required execution and inspection are done.
