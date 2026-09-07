from datasets import load_dataset

_dataset_cache: dict[str, object] = {}


def load_multi_news_split(split: str = "train"):
    """
    Load (and process-cache) a Multi-News split.

    load_single_cluster() used to call load_dataset() on every single call,
    which is fine for a one-off POC run but reloads the whole split on
    every iteration of a batch job (Track 2's generate_training_data.py
    loops over hundreds of clusters). Callers that need many clusters
    should load the dataset once with this and pass it into
    load_single_cluster(..., dataset=...).
    """
    if split not in _dataset_cache:
        print(f"Loading Multi-News dataset (split={split})...")
        _dataset_cache[split] = load_dataset("multi_news", split=split, trust_remote_code=True)
    return _dataset_cache[split]


def load_single_cluster(cluster_idx=0, dataset=None):
    """
    Load one document cluster from Multi-News test set.

    Args:
        cluster_idx: index of the cluster to load.
        dataset: an already-loaded HF dataset (from load_multi_news_split()).
            If None, loads (and caches) the test split — unchanged behavior
            for existing callers.
    """
    if dataset is None:
        dataset = load_multi_news_split("test")

    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")  # Multi-News separates docs with |||||
    reference_summary = cluster["summary"]

    print(f"\n{'='*80}")
    print(f"Loaded cluster {cluster_idx}")
    print(f"Number of documents: {len(documents)}")
    print(f"Reference summary length: {len(reference_summary)} chars")
    print(f"{'='*80}\n")

    return documents, reference_summary