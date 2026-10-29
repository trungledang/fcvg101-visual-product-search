"""Create a local private browsing page for the deterministic diagnostic examples."""
import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'reports/examples_manifest.json').read_text())
metrics = json.loads((root / 'reports/metrics.json').read_text())['main_official_split']
count = sum(len(rows) for rows in manifest['groups'].values() if isinstance(rows, list))
parts = [f'<!doctype html><meta charset="utf-8"><title>CLIP retrieval — {count} cases</title>',
         '<style>body{font:16px system-ui;max-width:1500px;margin:28px auto;padding:12px}img{width:100%;height:auto}article{margin:32px 0;border-top:1px solid #ddd}code{font-size:13px}</style>',
         '<h1>DeepFashion · frozen CLIP ViT-B/32</h1>',
         f'<p>Recall@1 {metrics["R@1"]:.2%}; Recall@5 {metrics["R@5"]:.2%}. These {count} outcome-balanced examples are diagnostic, not an estimate of error prevalence. Green: correct product ID; red: incorrect ID; blue: best known positive. Cosine similarity is not a probability.</p>']
for group, rows in manifest['groups'].items():
    if not isinstance(rows, list):
        continue
    parts.append('<h2>' + html.escape(group) + '</h2>')
    for row in rows:
        parts.append('<article><h3>' + html.escape(row['item_id']) + ' · first positive rank ' + str(row['first_pos_rank']) + '</h3><code>' + html.escape(row['path']) + '</code><br><a href="' + row['file'] + '"><img loading="lazy" src="' + row['file'] + '"></a></article>')
(root / 'artifacts/examples/index.html').write_text('\n'.join(parts), encoding='utf-8')
print(root / 'artifacts/examples/index.html')
