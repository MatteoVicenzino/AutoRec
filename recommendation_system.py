import torch
import numpy as np
import pandas as pd
from collections import Counter


class BookRecommendationSystem:

    def __init__(self, model_path, ids, k=300, df_books=None):
        self.k = k
        self.ids = ids
        self.df_books = df_books
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from Recomandation_models import FAE

        self.model = FAE(k=k, ids=ids)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Modello FAE caricato su {self.device}")
        print(f"✅ Vettore utente dimensione: {k}")
        print(f"✅ Book IDs nel modello: {len(ids)}")

    def get_recommendations(self, sparse_user, top_k=10):
        user_vector = self._sparse_to_tensor(sparse_user)

        with torch.no_grad():
            user_vector = user_vector.to(self.device)
            reconstruction, latent = self.model(user_vector, force_values=True)

        recommendations = self._find_new_books(
            original=user_vector.cpu(),
            reconstructed=reconstruction.cpu(),
            top_k=top_k
        )

        return recommendations

    def recommend_for_user_id(self, user_id, sparse_users_dict, top_k=10):
        if user_id not in sparse_users_dict:
            print(f"❌ Utente {user_id} non trovato nel dataset")
            return None

        sparse_user = sparse_users_dict[user_id]
        print(f"🔍 Processando utente {user_id} con {len(sparse_user)} rating")

        recommendations = self.get_recommendations(sparse_user, top_k)

        if recommendations:
            print(f"✅ Trovate {len(recommendations)} raccomandazioni")
        else:
            print("⚠️  Nessuna nuova raccomandazione trovata")

        return recommendations

    def _sparse_to_tensor(self, sparse_user):
        embedded = self._embed_fast_single(sparse_user, target_len=self.k)

        user_tensor = torch.zeros(1, self.k, 2)
        for i, (book_id, rating) in enumerate(embedded):
            if i >= self.k:
                break
            user_tensor[0, i, 0] = book_id
            user_tensor[0, i, 1] = rating

        return user_tensor

    def _embed_fast_single(self, sparse_user, target_len=300):
        embedding = [[book_id, rating] for book_id, rating, _ in sparse_user]
        existing_ids = set(book_id for book_id, _ in embedding)

        cluster_counts = Counter([cluster for _, _, cluster in sparse_user])
        total = len(sparse_user)
        cluster_ratios = {cluster: count / total for cluster, count in cluster_counts.items()}

        sorted_clusters = sorted(cluster_ratios.items(), key=lambda x: x[1], reverse=True)

        remaining = target_len - len(embedding)

        for cluster, ratio in sorted_clusters:
            if remaining <= 0:
                break

            n_to_sample = max(1, int(remaining * ratio))
            cluster_books = [book_id for book_id in self.ids
                             if book_id not in existing_ids]

            if cluster_books:
                sample_size = min(n_to_sample, len(cluster_books))
                sampled = np.random.choice(cluster_books, size=sample_size, replace=False)

                for book_id in sampled:
                    if len(embedding) >= target_len:
                        break
                    embedding.append([int(book_id), 0])
                    existing_ids.add(book_id)

                remaining -= sample_size

        while len(embedding) < target_len:
            embedding.append([0, 0])

        return embedding[:target_len]

    def _find_new_books(self, original, reconstructed, top_k):
        original_books = set()
        for k in range(original.shape[1]):
            book_id = int(original[0, k, 0].item())
            if book_id != 0:
                original_books.add(book_id)

        new_books = []
        for k in range(reconstructed.shape[1]):
            book_id = int(reconstructed[0, k, 0].item())
            rating = float(reconstructed[0, k, 1].item())

            if (book_id != 0 and
                    book_id not in original_books and
                    1.0 <= rating <= 5.0):
                new_books.append((book_id, rating))

        unique_books = {}
        for book_id, rating in new_books:
            if book_id not in unique_books or rating > unique_books[book_id]:
                unique_books[book_id] = rating

        new_books_sorted = sorted(unique_books.items(), key=lambda x: x[1], reverse=True)
        return new_books_sorted[:top_k]

    def print_recommendations(self, recommendations, show_details=True):
        if not recommendations:
            print("📚 Nessuna raccomandazione trovata")
            return

        print("📚 RACCOMANDAZIONI PERSONALIZZATE:")
        print("=" * 60)

        for i, (book_id, rating) in enumerate(recommendations, 1):
            if show_details and self.df_books is not None:
                book_info = self.df_books[self.df_books['goodreads_book_id'] == book_id]
                if not book_info.empty:
                    title = book_info.iloc[0]['title']
                    author = book_info.iloc[0]['authors']
                    avg_rating = book_info.iloc[0].get('average_rating', 'N/A')

                    print(f"{i:2d}. 📖 {title}")
                    print(f"    👤 Autore: {author}")
                    print(f"    ⭐ Rating predetto: {rating:.2f} | Media Goodreads: {avg_rating}")
                    print(f"    🆔 Book ID: {book_id}")
                else:
                    print(f"{i:2d}. 🆔 Book ID: {book_id} | ⭐ Rating: {rating:.2f}")
            else:
                print(f"{i:2d}. 🆔 Book ID: {book_id} | ⭐ Rating predetto: {rating:.2f}")
            print()


def create_recommendation_system(model_path='best_model.pth', ids=None, k=300, df_books=None):
    if ids is None:
        ids = list(range(1, 10001))
        print("⚠️  Usando book_ids di default. Sostituisci con gli IDs reali del training!")

    return BookRecommendationSystem(
        model_path=model_path,
        ids=ids,
        k=k,
        df_books=df_books
    )