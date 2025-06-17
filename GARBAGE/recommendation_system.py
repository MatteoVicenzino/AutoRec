import torch
import numpy as np
import pandas as pd
from collections import Counter


class BookRecommendationSystem:
    def __init__(self, model_path, book_ids, k=None, df_books=None):
        self.book_ids = book_ids
        self.k = k if k is not None else len(book_ids)
        self.df_books = df_books
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from Recomandation_models import FAE

        self.model = FAE(k=self.k, ids=book_ids)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.book_to_idx = {book_id: idx for idx, book_id in enumerate(book_ids)}
        self.idx_to_book = {idx: book_id for idx, book_id in enumerate(book_ids)}

        print(f"✅ Modello FAE caricato su {self.device}")
        print(f"✅ Vettore utente dimensione: {self.k}")
        print(f"✅ Book IDs nel modello: {len(book_ids)}")

    def _sparse_to_tensor(self, sparse_user):
        book_ids_tensor = torch.tensor(self.book_ids, dtype=torch.float32)
        ratings_tensor = torch.zeros(self.k, dtype=torch.float32)

        if isinstance(sparse_user, dict):
            for book_id, rating in sparse_user.items():
                if book_id in self.book_to_idx:
                    idx = self.book_to_idx[book_id]
                    ratings_tensor[idx] = float(rating)
        elif isinstance(sparse_user, list):
            for item in sparse_user:
                book_id = item[0]
                rating = item[1]
                if book_id in self.book_to_idx:
                    idx = self.book_to_idx[book_id]
                    ratings_tensor[idx] = float(rating)

        user_tensor = torch.stack([book_ids_tensor, ratings_tensor], dim=1)

        return user_tensor.unsqueeze(0)

    def _find_new_books(self, original, reconstructed, top_k=10):
        original_ratings = original.squeeze()[..., 1].numpy()  # (k,)
        predicted_ratings = reconstructed.squeeze()[..., 1].numpy()  # (k,)

        recommendations = []

        for idx, (orig_rating, pred_rating) in enumerate(zip(original_ratings, predicted_ratings)):
            if orig_rating == 0 and pred_rating > 0:
                book_id = self.book_ids[idx]  # Usa direttamente l'idx della lista book_ids
                recommendations.append((book_id, float(pred_rating)))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

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
            print("⚠️ Nessuna nuova raccomandazione trovata")

        return recommendations

    def get_book_info(self, book_id):
        if self.df_books is None:
            return {
                'book_id': book_id,
                'title': f'Book {book_id}',
                'authors': 'Unknown',
                'average_rating': 'N/A',
                'ratings_count': 'N/A'
            }

        book_info = self.df_books[self.df_books['goodreads_book_id'] == book_id]

        if len(book_info) > 0:
            book = book_info.iloc[0]
            return {
                'book_id': book_id,
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book['ratings_count'],
                'publication_year': book.get('original_publication_year', 'N/A'),
                'language': book.get('language_code', 'N/A')
            }
        else:
            return {
                'book_id': book_id,
                'title': f'Book {book_id}',
                'authors': 'Unknown',
                'average_rating': 'N/A',
                'ratings_count': 'N/A'
            }

    def display_recommendations(self, user_ratings, top_k=5, show_user_books=True):
        print("🎯 RACCOMANDAZIONI FAE")
        print("=" * 50)

        if isinstance(user_ratings, dict):
            ratings_dict = user_ratings
            rated_count = len(user_ratings)
        elif isinstance(user_ratings, list):
            ratings_dict = {item[0]: item[1] for item in user_ratings}
            rated_count = len(user_ratings)
        else:
            ratings_dict = {}
            rated_count = 0

        if show_user_books and ratings_dict:
            print(f"\n📚 Libri che hai valutato ({rated_count} totali):")
            displayed_books = 0
            for book_id, rating in list(ratings_dict.items())[:5]:
                book_info = self.get_book_info(book_id)
                if book_info:
                    print(f"   ⭐ {rating}/5 - {book_info['title']} ({book_info['authors']})")
                    displayed_books += 1

            if rated_count > 5:
                print(f"   ... e altri {rated_count - displayed_books} libri")

        recommendations = self.get_recommendations(user_ratings, top_k)

        if not recommendations:
            print("\n❌ Impossibile generare raccomandazioni")
            print("💡 Suggerimenti:")
            print("   - Verifica che i libri valutati siano nel dataset")
            print("   - Prova ad aggiungere più valutazioni")
            return

        print(f"\n🎁 Top {len(recommendations)} raccomandazioni per te:")
        print("-" * 50)

        for i, (book_id, predicted_rating) in enumerate(recommendations, 1):
            book_info = self.get_book_info(book_id)
            print(f"\n{i}. 📖 {book_info['title']}")
            print(f"   ✍️  {book_info['authors']}")
            print(f"   🔮 Rating predetto: {predicted_rating:.2f}/5")

            if book_info['average_rating'] != 'N/A':
                print(f"   ⭐ Rating medio: {book_info['average_rating']:.2f}/5")
            if book_info['ratings_count'] != 'N/A':
                print(f"   👥 Valutazioni: {book_info['ratings_count']:,}")
            if book_info.get('publication_year', 'N/A') != 'N/A':
                try:
                    year = int(float(book_info['publication_year']))
                    print(f"   📅 Anno: {year}")
                except:
                    pass

    def get_user_vector_info(self, sparse_user):
        user_tensor = self._sparse_to_tensor(sparse_user)
        ratings = user_tensor.squeeze()[..., 1].numpy()  # Estrae solo i rating

        non_zero_count = np.count_nonzero(ratings)
        sparsity = 1 - (non_zero_count / len(ratings))

        if non_zero_count > 0:
            non_zero_ratings = ratings[ratings > 0]
            avg_rating = np.mean(non_zero_ratings)
            rating_dist = Counter(non_zero_ratings.astype(int))
        else:
            avg_rating = 0
            rating_dist = {}

        stats = {
            'total_books': len(ratings),
            'rated_books': non_zero_count,
            'sparsity': sparsity,
            'average_rating': avg_rating,
            'rating_distribution': dict(rating_dist)
        }

        return stats

    def batch_recommendations(self, users_dict, top_k=10, verbose=True):
        batch_results = {}
        total_users = len(users_dict)

        for i, (user_id, sparse_user) in enumerate(users_dict.items(), 1):
            if verbose and i % 100 == 0:
                print(f"📊 Processati {i}/{total_users} utenti...")

            recommendations = self.get_recommendations(sparse_user, top_k)
            batch_results[user_id] = recommendations

        if verbose:
            print(f"✅ Completate raccomandazioni per {total_users} utenti")

        return batch_results

    def evaluate_recommendations(self, test_users_dict, top_k=10):
        total_users = len(test_users_dict)
        successful_recs = 0
        total_recommendations = 0

        for user_id, sparse_user in test_users_dict.items():
            recommendations = self.get_recommendations(sparse_user, top_k)

            if recommendations:
                successful_recs += 1
                total_recommendations += len(recommendations)

        coverage = successful_recs / total_users if total_users > 0 else 0
        avg_recs_per_user = total_recommendations / total_users if total_users > 0 else 0

        metrics = {
            'user_coverage': coverage,
            'avg_recommendations_per_user': avg_recs_per_user,
            'total_users_evaluated': total_users,
            'users_with_recommendations': successful_recs
        }

        print("📊 METRICHE DI VALUTAZIONE")
        print("=" * 30)
        print(f"👥 Utenti valutati: {metrics['total_users_evaluated']:,}")
        print(f"✅ Utenti con raccomandazioni: {metrics['users_with_recommendations']:,}")
        print(f"📈 Copertura utenti: {metrics['user_coverage']:.2%}")
        print(f"📊 Raccomandazioni medie per utente: {metrics['avg_recommendations_per_user']:.1f}")

        return metrics


def load_recommendation_system(model_path, book_ids, books_csv_path=None):
    print("🚀 Caricamento sistema di raccomandazione...")

    k = len(book_ids)

    df_books = None
    if books_csv_path:
        try:
            df_books = pd.read_csv(books_csv_path)
            print(f"📚 Caricato dataset libri: {len(df_books):,} libri")
        except Exception as e:
            print(f"⚠️ Errore nel caricamento {books_csv_path}: {e}")

    system = BookRecommendationSystem(
        model_path=model_path,
        book_ids=book_ids,
        k=k,
        df_books=df_books
    )

    print("✅ Sistema di raccomandazione pronto!")
    return system