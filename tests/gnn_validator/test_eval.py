from gnn_validator.eval import auroc, f1_at_threshold


def test_auroc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0.0, 0.0, 1.0, 1.0]
    assert auroc(scores, labels) == 1.0


def test_auroc_worst_case():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0.0, 0.0, 1.0, 1.0]
    assert auroc(scores, labels) == 0.0


def test_auroc_matches_known_value_with_ties():
    # 3 negatives, 2 positives, one tie between a positive and a negative.
    # By hand: pairs sorted -> (0.1,0) (0.3,0) (0.5,1) (0.5,0) (0.9,1)
    # ranks: 1, 2, 3.5, 3.5, 5 ; rank_sum_pos = 3.5 + 5 = 8.5
    # U = 8.5 - 2*3/2 = 5.5 ; AUROC = 5.5 / (2*3) = 0.9166...
    scores = [0.1, 0.3, 0.5, 0.5, 0.9]
    labels = [0.0, 0.0, 1.0, 0.0, 1.0]
    assert abs(auroc(scores, labels) - (5.5 / 6)) < 1e-9


def test_auroc_undefined_without_both_classes():
    assert auroc([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) is None
    assert auroc([0.1, 0.2, 0.3], [0.0, 0.0, 0.0]) is None


def test_f1_at_threshold_basic():
    scores = [0.9, 0.8, 0.4, 0.1]
    labels = [1.0, 0.0, 1.0, 0.0]
    # tp=1 (0.9), fp=1 (0.8), fn=1 (0.4) -> precision=0.5, recall=0.5, f1=0.5
    assert abs(f1_at_threshold(scores, labels, 0.5) - 0.5) < 1e-9


def test_f1_at_threshold_no_positive_predictions():
    assert f1_at_threshold([0.1, 0.2], [1.0, 0.0], threshold=0.9) == 0.0
