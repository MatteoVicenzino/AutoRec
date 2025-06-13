# AutoencoderRecommander
Developing a Collaborative Recommender System using Autoencoders
We will use the https://www.kaggle.com/datasets/alexanderfrosati/goodbooks-10k-updated?resource=download  
IDEA:
The process is based on the assumption that user preferences are highly correlated with each other. First, we structure the data as a matrix with the users on the rows and the books as columns.
Then we train an Autoencoder using the users, vector of ratings for all books as data unit
The recomandation is made by passing a user through the Autoencoder and picking the new item appearing in the reconstruction.

1) FILTERING:
The first problem is the dimension of the user-item matrix and its sparsity, infact in this case we have a matrix 53424x10000 with at maximum 8 non zero values for row.
The second problem with this technique is the dimension of the data; adding a single book to the list means changing the dimension of every data and also need to change the structure of the model.
To solve this problem we thought it will be necessary a filtering on the user matrix. 
This will be done by doing a clustering on the books using some features. Then, for every book a user like, find a number of similar books, books from the same cluster. The number of all the books retrived from this part need to be equal to N that will be the dimension of the data.
This way having N << #total of books we can solve booth the problems. Adding a book, until a certin number of addition, can simply be solved by putting new books in the most similar cluster.  