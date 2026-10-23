# FCVG101: Visual Product Search feasibility pilot

Status: repository prepared; no experiment has been run and no retrieval metrics exist yet.

Purpose: determine whether Visual Product Search is worth pursuing for FCVG101. This is a feasibility experiment, not the coursework submission.

## Start in Claude Code Cloud

Select this repository at https://claude.ai/code and submit:

> Execute PILOT_TASK.md. Implement and run the feasibility pilot, starting with dataset access and compute preflight. Follow CLAUDE.md. Report actual measurements only. If access requires my login or dataset authorization, ask for the exact action while continuing independent implementation. Write the final concise report in Vietnamese.

Cloud must be able to reach the dataset host, Python package registry and pretrained checkpoint host. Inspect the actual CPU/GPU, RAM and disk before choosing a run size. GitHub access does not imply dataset access or GPU availability.

## Dataset

Official benchmark: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

Original dataset terms/access: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

Use the original **In-shop Clothes Retrieval** release, not DeepFashion2, the category/attribute subset, or the consumer-to-shop benchmark. Keep raw data and model weights outside git. A private repository is not permission to redistribute dataset images. Store visual examples privately in the execution environment unless redistribution permission is verified; keep their manifest and access instructions in the report.

## Expected outputs from the cloud run

- Reproducible baseline code and exact environment/commands.
- Dataset/split audit, metrics JSON, per-query ranks and resource measurements.
- At least 20 retrieval examples spanning successes and failures, with an example-selection manifest.
- Concise Vietnamese report with actual results, limitations, costs, failure analysis and Go / Change Dataset / Change Topic recommendation.

See PILOT_TASK.md for the complete experiment protocol. No training or web interface is in scope.
