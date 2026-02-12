from datasets import load_dataset


def load_single_cluster(cluster_idx=0):
    """Load one document cluster from Multi-News test set."""
    print("Loading Multi-News dataset...")
    dataset = load_dataset("multi_news", split="test", trust_remote_code=True)

    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")  # Multi-News separates docs with |||||
    reference_summary = cluster["summary"]

    print(f"\n{'='*80}")
    print(f"Loaded cluster {cluster_idx}")
    print(f"Number of documents: {len(documents)}")
    print(f"Reference summary length: {len(reference_summary)} chars")
    print(f"{'='*80}\n")

    return documents, reference_summary