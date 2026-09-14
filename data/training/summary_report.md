# Training data summary report (Track 2.5)

## Overview

- Clean KGs: 13
- Train split: 4 clusters
- Val split: 1 clusters
- Test split: 0 clusters

## Class balance (corrupted variants per corruption type)

| Corruption type | Count |
|---|---|
| contradictions | 39 |
| fragmentation | 39 |
| missing_entities | 39 |

Corruption types are reasonably balanced (all within 80% of the max).

## Counts per split x corruption type

| Split | contradictions | fragmentation | missing_entities |
|---|---|---|---|
| train | 12 | 12 | 12 |
| val | 3 | 3 | 3 |
| test | 0 | 0 | 0 |

## Structural validity

✓ No problems found — every corrupted graph has >=1 entity, no relation references a missing entity id.

## Spot-check sample

- Cluster 6, **contradictions** (severity=0.1): 30e/10r → 30e/10r — {'corruption_type': 'contradictions', 'severity': 0.1, 'flipped_relations': 1}
- Cluster 11, **fragmentation** (severity=0.3): 30e/22r → 30e/15r — {'corruption_type': 'fragmentation', 'severity': 0.3, 'removed_relations': 7}
- Cluster 0, **fragmentation** (severity=0.1): 47e/48r → 47e/43r — {'corruption_type': 'fragmentation', 'severity': 0.1, 'removed_relations': 5}
- Cluster 7, **fragmentation** (severity=0.2): 31e/22r → 31e/18r — {'corruption_type': 'fragmentation', 'severity': 0.2, 'removed_relations': 4}
- Cluster 13, **missing_entities** (severity=0.3): 11e/5r → 8e/0r — {'corruption_type': 'missing_entities', 'severity': 0.3, 'removed_entities': 3, 'removed_relations': 5}
- Cluster 13, **fragmentation** (severity=0.2): 11e/5r → 11e/4r — {'corruption_type': 'fragmentation', 'severity': 0.2, 'removed_relations': 1}
- Cluster 13, **contradictions** (severity=0.2): 11e/5r → 11e/5r — {'corruption_type': 'contradictions', 'severity': 0.2, 'flipped_relations': 1}
- Cluster 11, **missing_entities** (severity=0.3): 30e/22r → 21e/4r — {'corruption_type': 'missing_entities', 'severity': 0.3, 'removed_entities': 9, 'removed_relations': 18}
