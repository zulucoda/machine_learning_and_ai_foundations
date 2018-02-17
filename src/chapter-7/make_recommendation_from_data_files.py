import numpy
import pandas
import pickle

# load predicted data rules from files
U = pickle.load(open('user_features.dat', 'rb'))
M = pickle.load(open('product_features.dat', 'rb'))
predicted_ratings = pickle.load(open('predicted_ratings.dat', 'rb'))

# load movies list
movies_list = pandas.read_csv('../chapter-4/movies.csv', index_col='movie_id')

print('Enter a user_id to get recommendations (Between 1 to 100):')
user_id_to_search = int(input())

print('Movies we will recommend:')
user_ratings = predicted_ratings[user_id_to_search - 1]
movies_list['rating'] = user_ratings
movies_list = movies_list.sort_values(by=['rating'], ascending=False)

print(movies_list[['title', 'genre', 'rating']].head(10))