import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import multiprocessing as mp
from functools import partial
from tqdm.notebook import tqdm  # per barra di progresso
import torch  # se vuoi direttamente tensori

# embedding globale (non passata via pickle)
global_df_books = None

def init_globals(df_books):
    global global_df_books
    global_df_books = df_books

def find_clusters_ratio(sparse_user):
    cluster_counts = Counter([cluster for _, _, cluster in sparse_user])
    total = len(sparse_user)
    cluster_ratios = {cluster: (count / total, count) for cluster, count in cluster_counts.items()}
    return sorted([[cluster, ratio, count] for cluster, (ratio, count) in cluster_ratios.items()],
                  key=lambda x: x[1], reverse=True)

def sample_from_cluster_fast(n, cluster, existing_ids):
    cluster_books = global_df_books[global_df_books['cluster'] == cluster]
    cluster_books = cluster_books[~cluster_books['goodreads_book_id'].isin(existing_ids)]
    
    if len(cluster_books) == 0:
        return []

    sample_n = min(n, len(cluster_books))
    samples = cluster_books.sample(sample_n)
    return [[row['goodreads_book_id'], 0] for _, row in samples.iterrows()]

def embed_fast_single(sparse_user, target_len=300):
    embedding = [[book_id, rating] for book_id, rating, _ in sparse_user]
    existing_ids = set(book_id for book_id, _ in embedding)
    
    user_clusters = find_clusters_ratio(sparse_user)
    remaining = target_len - len(embedding)

    for cluster, ratio, _ in user_clusters:
        n = int(remaining * ratio)
        new_samples = sample_from_cluster_fast(n, cluster, existing_ids)
        embedding.extend(new_samples)
        existing_ids.update([book_id for book_id, _ in new_samples])

    while len(embedding) < target_len:
        cluster = user_clusters[0][0]
        new_samples = sample_from_cluster_fast(1, cluster, existing_ids)
        if not new_samples:
            # fallback: prendo un libro random da tutto il df_books
            global global_df_books
            remaining_needed = target_len - len(embedding)
            all_books = global_df_books[~global_df_books['goodreads_book_id'].isin(existing_ids)]
            if len(all_books) == 0:
                print(f"No more books available anywhere to fill up. Final length: {len(embedding)}")
                break
            sample_n = min(remaining_needed, len(all_books))
            samples = all_books.sample(sample_n)
            new_samples = [[row['goodreads_book_id'], 0] for _, row in samples.iterrows()]
            embedding.extend(new_samples)
            existing_ids.update([book_id for book_id, _ in new_samples])
            break
        embedding.extend(new_samples)
        existing_ids.add(new_samples[0][0])
    
    print(f"Final embedding length: {len(embedding)}")
    return torch.tensor(embedding)

def embed_all_users(sparse_users: dict, df_books, num_processes=8):
    init_args = (df_books,)
    with mp.Pool(processes=num_processes, initializer=init_globals, initargs=init_args) as pool:
        users_data = list(sparse_users.values())
        result = list(tqdm(pool.imap(embed_fast_single, users_data), total=len(users_data)))
    return result
