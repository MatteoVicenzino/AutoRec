RECOMMENDATION SYSTEM USING AUTOENCODER                               
Dataset: https://www.kaggle.com/datasets/alexanderfrosati/goodbooks-10k-updated?resource=download             
           
IDEA:            
The process is based on two main assumptions:                    
1-A user's preferences are highly correlated and structured.                
2-An autoencoder can effectively learn these underlying associations.                            
Each user is represented by a binary vector whose length equals the number of items, where a 1 indicates interest in a specific item, and 0 otherwise.                
The autoencoder is trained to reconstruct these vectors, implicitly learning the spatial (latent) relationships between items.                              
If training is successful, the reconstruction will closely match the original input, with discrepancies occurring only in a few positions. These differing positions can then be interpreted as potential new items that the  user might be interested in.           
                 
PREPROCESSING:                             
First, we need to prepare the dataset by removing missing values (NaNs) and merging the necessary information into a single file.                     
This preprocessing is handled in the notebook DataPreparation.ipynb, which generates the file books_autorec.csv.                
The next step is the creation of the user-item matrix:                      
1- Data-Filtering: To speed up execution and demonstrate the model’s functionality, we selected the top 2,000 books ranked by number of ratings, and the top 50,000 users ranked by the number of ratings they gave to these                       selected books. This user selection strategy aims to reduce the sparsity of the resulting user-item matrix. This process generates filter_users, a python dictionary containing user_id as keys and book_id and ratings as values.         
2- Mapping: This part builds two dictionaries: one that maps each book_id to its corresponding column index in the user-item matrix, and another that performs the reverse mapping.  
3- Matrix: Then, we finally build the matrix and store it in a pd.dataframe, calling it df_input_data.    
       
MODEL:     
The model, called Spatial_F_AE, is a deep fully connected autoencoder, and it is trained using the Mean Squared Error (MSE) loss function. The model takes as input a binary user vector, where each position contains either a 1 or a 0, indicating whether the user liked a specific item. The output is a reconstructed vector with values between 0 and 1. Since a sigmoid activation function is used at the output layer, these values can be interpreted as probabilities that the user would like each corresponding item.   

NEW USER:      
To address the cold-start problem with new users, we propose a clustering-based approach.     
Books are clustered using K-means, based on selected characteristics such as author, publication year, and especially tags.      
To process the tags, we use TfidfVectorizer from sklearn.feature_extraction.text, which allows us to encode the textual information into numerical vectors and compute the semantic similarity between them.      
The quality of the clustering is evaluated by inspecting the most frequent tags within each cluster to verify coherence.      
From each cluster, a small number of representative books (champions) is selected and presented to the user. The champions are selected as the most popular in the cluster. Based on the user's preferences, we construct a personalized user vector, marking liked champions with a 1. This enables the model to generate recommendations even in the absence of historical user data.  
Additionally, we do not retrain the model when a new user is added. This is based on the assumption that for a small number of new users, the model has already learned a sufficiently general representation of user behavior.    

EVALUATION:   
The model is evaluated using Recall@K, a metric commonly used in recommender systems.       
For each user, we simulate a realistic recommendation scenario by removing a portion of the items they liked (setting those entries in the user vector to 0).     
The modified user vector is then passed through the model to generate recommendations.      
We measure performance by calculating the proportion of the removed items that appear in the top-K recommendations produced by the model. This process is repeated for all users, and we compute the average recall to obtain an overall evaluation of the model's ability to recover relevant items.   

CONCLUSION:
The model demonstrates good performance based on the Recall@K metric, providing accurate and meaningful recommendations. Its inference is also computationally efficient, since it only requires passing user vectors through the network and selecting the top-K items from the reconstructed output.      
However, there are several limitations to consider:     
1-Large Datasets: When the dataset grows significantly, the user-item matrix becomes harder to manage due to its size, leading to increased memory usage and slower training and inference.       
2-Scalability Issues: If adding users is manageable, as new users can be represented by a binary preference vector without retraining the model. Adding Items presents a major limitation. Since the model architecture is tied to the dimensionality of the user vectors (i.e., the number of items), adding new items requires a complete retraining of the model. Otherwise, the new vectors would no longer be compatible with the existing model weights.
