# A-Hybrid-approach-to-Movie-Recommendation-System-using-Machine-Learning
Our input data set is originally from the online movie critique giant ”Movie Lens (TMDB)”. The objective of our data set is to provide data that is helpful in analyzing and recommending movies to the user that the user may be interested in and has a higher chance of viewing, generating revenue for any online streaming platform. This dataset consists of the following files:

i. movies metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

ii. keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

iii. credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

iv. links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

v. links small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset. 

vi. ratings small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies. 

The recommendation system machine learning techniques which we are using are as follows:

1. Popularity-Based Filtering: This is one of the first recommendation systems to ever get developed and successfully deployed. As its name suggests, it has a working principle based on popularity or the latest trend. These systems check the movies which are most popular among the users and directly recommend those without considering other factors like user history i.e. what movies the user has viewed or interacted with in the past, etc. This has a chance to increase user engagement when compared to no recommendation system. But such systems have a huge drawback that they do not provide personalized recommendations to users.
                                                
                                                Weighted Rank (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
where,
R = average for the movie (mean) = (Rating)
v = number of votes for the movie = (votes)
m = minimum votes required to be listed in the Top 250
C = the mean vote across the whole report 

2.	Content-Based Filtering: Content-Based filtering recommendation system tries to make suggestions on attributes of items that a customer previously liked or interacted with. Such a system first builds item profiles and then derives a user profile from them. This user profile contains features that are important for a particular individual. Once the system knows about a customer's preferences, it can recommend new items according to it. This technique considers a wide variety of factors which include both objective and descriptive factors before making any recommendations. 

For content-based filtering, we have used the TF-IDF (Term Frequency Inverse Document Frequency) technique, which is used to transform the text into a meaningful representation of numbers that is used to fit machine learning algorithms.  

A. Term Frequency is the number of times a term appears in a particular document. So it’s specific to a document. It is calculated as:
                          
                          TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

B. Inverse Document Frequency is a measure of how common or rare a term appears across the entire corpus of documents. So it is common to all documents. If the word is common and appears in many documents, the idf value (normalized) will approach 0 or else approach 1 if it’s rare. 
IDF is calculated as:  

                          IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

3. Collaborative filtering: Collaborative filtering recommendation systems first look at interactions between users and items try to find either similar types of users or similar types of products and based on this similarity, calculate prediction scores and then recommend new products to target users. Every user and item is described by a feature vector in the same space. Collaborative Filtering is mainly of two types:

a. Item-Item based Collaborative Filtering: This technique starts by searching the items that the user has interacted with previously, and finds similar items to it with the help of similarity metrics, and using which a prediction function is defined which then suggests and recommends items to the user. The steps involved in Item-Item based Collaborative Filtering are as follows:

Step-1: First, convert the given data in the form of a user-item matrix.
Step-2: We then start building the model by finding similarity between all the item pairs. We create vectors for each individual item using the user ratings that are known and then we find the similarity between them which can be found in multiple ways but we have used cosine similarity in our framework.

Cosine Similarity: Cosine similarity is a metric used to measure the similarity of two vectors which is measured in terms of the cosine of the angle between the two vectors . Specifically, it measures the similarity in the direction or orientation of the vectors irrespective of their magnitude or scale. Both vectors need to lie on the same plane in the same inner product space, meaning they must produce a scalar through dot product multiplication. Mathematically, the cosine similarity of two vectors A and B is defined as the dot product of the vectors divided by their magnitude as follows:

![image](https://user-images.githubusercontent.com/88252622/230611729-1c1ba1ca-3ad7-4422-8ed5-4209187a1541.png)

Step-3: Calculate the recommendation or rating score as: 

![gif](https://user-images.githubusercontent.com/88252622/230612256-ce79d9f2-3bfa-48d5-ad1c-15d6f0e6657e.gif)

b. User-User based Collaborative Filtering: This technique looks for similar users i.e. users with similar interests, based on the items the users have already rated or interacted with, and recommends other items that may be of interest to other target users. The steps involved in User-User based Collaborative Filtering are as follows:

Step-1: Convert the given data in the form of a user-item matrix based on the interest of a user in a certain item.
Step-2: We then start building the model by finding the set of similar users to the target user for which we have again used the cosine similarity according to the user-item matrix. 
Step-3: Calculate the recommendation score as: 

![gif](https://user-images.githubusercontent.com/88252622/230612331-5ca87523-de59-4176-8b92-8abec0c31727.gif)

In order for both of these collaborative filtering methods to work, we have used the Singular Value Decomposition (SVD) technique, which is basically a matrix factorization technique that decomposes any given matrix into 3 generic and familiar matrices. The SVD of a m X n matrix A is given by the formula:

![quicklatex com-c931b3a37227b539911c0837f16a540d_l3](https://user-images.githubusercontent.com/88252622/230612896-1017a31e-4c34-491f-b0e5-6f925864936a.svg)

![gif](https://user-images.githubusercontent.com/88252622/230614256-005a5d9c-36af-4a41-abc4-31a84c4b545a.gif)

where U and V are orthogonal matrices with orthonormal eigenvectors chosen from  AA^T and A^TA respectively. S is a diagonal matrix with r elements equal to the root of the positive eigenvalues of both matrices U and V, which have the same positive eigenvalues. The diagonal elements are composed of singular values.


4. Hybrid Filtering: This approach combines the previously mentioned algorithms and then makes recommendations to the user. This algorithm is generally preferred and used by the major leaders of online streaming services since it has the advantages of all the other techniques.   
