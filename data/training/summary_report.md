# Training data summary report (Track 2.5)

## Overview

- Clean KGs: 5
- Train split: 4 clusters
- Val split: 1 clusters
- Test split: 0 clusters

## Class balance (corrupted variants per corruption type)

| Corruption type | Count |
|---|---|
| contradictions | 15 |
| fragmentation | 15 |
| missing_entities | 15 |

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

- Cluster 4, **fragmentation** (severity=0.2): 45e/26r → 45e/21r — {'corruption_type': 'fragmentation', 'severity': 0.2, 'removed_relations': 5}
- Cluster 0, **missing_entities** (severity=0.2): 47e/48r → 38e/9r — {'corruption_type': 'missing_entities', 'severity': 0.2, 'removed_entities': 9, 'removed_relations': 39}
- Cluster 0, **contradictions** (severity=0.2): 47e/48r → 47e/48r — {'corruption_type': 'contradictions', 'severity': 0.2, 'flipped_relations': 10}
- Cluster 1, **missing_entities** (severity=0.3): 21e/16r → 15e/2r — {'corruption_type': 'missing_entities', 'severity': 0.3, 'removed_entities': 6, 'removed_relations': 14}
- Cluster 1, **missing_entities** (severity=0.1): 21e/16r → 19e/8r — {'corruption_type': 'missing_entities', 'severity': 0.1, 'removed_entities': 2, 'removed_relations': 8}
- Cluster 1, **fragmentation** (severity=0.3): 21e/16r → 21e/11r — {'corruption_type': 'fragmentation', 'severity': 0.3, 'removed_relations': 5}
- Cluster 0, **missing_entities** (severity=0.3): 47e/48r → 33e/3r — {'corruption_type': 'missing_entities', 'severity': 0.3, 'removed_entities': 14, 'removed_relations': 45}
- Cluster 0, **missing_entities** (severity=0.1): 47e/48r → 42e/14r — {'corruption_type': 'missing_entities', 'severity': 0.1, 'removed_entities': 5, 'removed_relations': 34}
