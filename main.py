import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

books = pd.read_csv('/Users/apple/Downloads/Books_dataset/input/book-recommendation-dataset/Books.csv')
ratings = pd.read_csv('/Users/apple/Downloads/Books_dataset/input/book-recommendation-dataset/Ratings.csv')
users = pd.read_csv('/Users/apple/Downloads/Books_dataset/input/book-recommendation-dataset/Users.csv')

book_ratings = ratings.merge(books, on='ISBN')

Rating_count = book_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
Rating_count.rename(columns={'Book-Rating': 'Rating Count'}, inplace=True)

AvgRating_count = book_ratings.groupby('Book-Title').mean()['Book-Rating'].reset_index()
AvgRating_count.rename(columns={'Book-Rating': 'Avg Rating'}, inplace=True)

popular_df = Rating_count.merge(AvgRating_count, on='Book-Title')

popular_df = popular_df[popular_df['Rating Count'] >= 250]

popular_df = popular_df.sort_values('Avg Rating', ascending=False).head(50)
users.drop('Age', axis=1, inplace=True)
books.dropna(inplace=True)
book_ratings.dropna(inplace=True)

x = book_ratings.groupby('User-ID').count()['Book-Rating'] > 200
exp_users = x[x].index
filtered_rating_users = book_ratings[book_ratings['User-ID'].isin(exp_users)]

y = filtered_rating_users.groupby('Book-Title').count()['Book-Rating'] > 50
famous_books = y[y].index
final_ratings = filtered_rating_users[filtered_rating_users['Book-Title'].isin(famous_books)]

final_df = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
final_df.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(final_df)


def recommend(book):
    try: 
        book_index = np.where(final_df.index == book)[0][0]
        distances = similarity_scores[book_index]
        print(list(enumerate(distances)))
        book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        print(book_list)
        for i in book_list:
            print(final_df.index[i[0]])
    except:
        print('Not enough reviews found on the item, consider the books below based on their popularity:')
        for i in final_df.head(5).index.values:
            print(i)

recommend('Lying Awake')
