# Project scope

Read PILOT_TASK.md before implementation. The user wants evidence for choosing a course topic, not a completed coursework project.

- Use frozen pretrained OpenAI CLIP ViT-B/32 and image-to-image cosine retrieval only for this baseline. No fine-tuning, additional trained model, or web interface.
- Work autonomously on reversible implementation and verification. Ask only for concrete missing access, credentials entered by the user, or spending decisions. Do not provision paid compute.
- Verify actual execution separately from code readiness. Never manufacture metrics, plots, costs or completion claims. Synthetic fixtures test correctness only and never support dataset conclusions.
- Keep full official gallery as distractors. A smaller, seeded query sample may be used when measured compute requires it, but label results partial and preserve all identities/categories in sampling.
- Follow official identities and splits. Same identity in query/gallery is required; the same image in both is leakage. Train identities must not overlap evaluation identities.
- Do not use test metrics to tune parameters or select a method. Improvement ideas are proposals for later work.
- Use repository-local dependencies/caches where practical; do not commit data, weights, secrets or unrelated course materials.
- Commit small reproducible code, manifests and aggregate results. Verify dataset redistribution terms before committing any images; otherwise provide a private artifact location and regeneration command.
- If dataset access blocks execution, finish independently useful code and tests, document the exact blocker and ask the user for the specific access step. Do not silently substitute another dataset or recommend changing topic solely because of a download problem.
