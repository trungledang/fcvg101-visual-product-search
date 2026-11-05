"""Independent metric check: rank best positive by counting better gallery scores."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT / 'cache/embeddings'
    Q = np.load(cache / 'query.npy')
    G = np.load(cache / 'gallery.npy')
    qpaths = (cache / 'query.paths.txt').read_text().splitlines()
    gpaths = (cache / 'gallery.paths.txt').read_text().splitlines()
    assert Q.shape == (14218,512) and G.shape == (12612,512)
    assert not set(qpaths) & set(gpaths)
    assert all(np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4) for x in (Q,G))
    qids = [p.split('/')[3] for p in qpaths]
    gids = np.array([p.split('/')[3] for p in gpaths])
    results = list(csv.DictReader((ROOT / 'reports/per_query_results.csv').open()))
    assert [r['query_path'] for r in results] == qpaths
    gallery_indices = np.arange(len(G))
    ranks = []
    for start in range(0,len(Q),128):
        sims = Q[start:start+128] @ G.T
        for i,scores in enumerate(sims):
            positives = np.flatnonzero(gids == qids[start+i])
            assert len(positives)
            best = positives[np.argmax(scores[positives])]
            ranks.append(1+int(np.sum(scores>scores[best]))+
                         int(np.sum((scores==scores[best]) & (gallery_indices<best))))
    expected = np.array([int(r['first_pos_rank']) for r in results])
    ranks = np.array(ranks)
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())['main_official_split']
    verified = {f'R@{k}':float(np.mean(ranks<=k)) for k in (1,5)}
    assert all(abs(verified[f'R@{k}']-metrics[f'R@{k}'])<1e-12 for k in (1,5))
    changed_hits = {f'K={k}':int(np.sum((ranks<=k)!=(expected<=k))) for k in (1,5)}
    assert not any(changed_hits.values()), changed_hits
    out = {'method':'all queries: independently count scores better than best positive; no sorting or retrieval helper',
           'queries':len(Q),'gallery':len(G),'metrics':verified,
           'rank_mismatches':int(np.sum(ranks!=expected)),
           'per_query_hit_classification_changes':changed_hits,
           'rank_difference_examples':[{'query':qpaths[i],'cloud_rank':int(expected[i]),'independent_rank':int(ranks[i])}
                                       for i in np.flatnonzero(ranks!=expected)],
           'headline_metrics_match':True}
    (ROOT / 'reports/independent_verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
