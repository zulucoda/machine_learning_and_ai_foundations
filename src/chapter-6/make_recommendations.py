import sys
sys.path.insert(0, '../chapter-5/')

import numpy
import pandas
import matrix_factorization_utilities

users_movie_ratings_list = pandas.read_csv('../chapter-4/movie_ratings_data_set.csv')

movies_list = pandas.read_csv('../chapter-4/movies.csv', index_col='movie_id')

# convert user movie ratings list into a matrix / pivot table
users_movie_ratings_pivot_table = pandas.pivot_table(users_movie_ratings_list, index='user_id', columns='movie_id',
                                                     aggfunc=numpy.max)

# apply matrix factorisation to find latent (hidden) features
U,M = matrix_factorization_utilities.low_rank_matrix_factorization(users_movie_ratings_pivot_table.as_matrix(),
                                                                   num_features=15,
                                                                   regularization_amount=1.0)

# find predicted ratings by multiplying U and M using numpy.matmul
predicted_ratings = numpy.matmul(U, M)

print('Enter a user_id to get recommendations (Between 1 and 100):')
user_id_to_search = int(input())

print('Movies previously reviewed by user_id {}:'.format(user_id_to_search))

reviewed_movies_list = users_movie_ratings_list[users_movie_ratings_list['user_id'] == user_id_to_search]
reviewed_movies_list = reviewed_movies_list.join(movies_list, on='movie_id')

print(reviewed_movies_list[['title','genre','value']])

input('Press enter to continue.')

print('Movies we will recommend:')

user_ratings = predicted_ratings[user_id_to_search - 1]
movies_list['rating'] = user_ratings

already_reviewed = reviewed_movies_list['movie_id']
recommended_list = movies_list[movies_list.index.isin(already_reviewed) == False]
recommended_list = recommended_list.sort_values(by=['rating'], ascending=False)

print(recommended_list[['title', 'genre', 'rating']].head(10))