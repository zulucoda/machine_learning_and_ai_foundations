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

# swap the rows and cols of product_features just so it's easier to work with
M = numpy.transpose(M)

print('Enter movie id:')
get_movie_id = input()
print(int(get_movie_id))
movie_id = int(get_movie_id)

selected_movie = movies_list.loc[movie_id]

print('We are finding movies similar to this movie:')
print('Movie title: {}'.format(selected_movie.title))
print('Genre: {}'.format(selected_movie.genre))

current_movie_features = M[movie_id - 1]
print('The attributes for this movie are:')
print(current_movie_features)

# Logic for finding similar movies:

# 1) Subtract the current movie features from every other movie feature
difference = M - current_movie_features
# 2) Take the absolute value difference (so all numbers are positive)
absolute_difference = numpy.abs(difference)
# 3) Each movie has 15 features. Sum 15 features to get a total 'difference score' for each
total_difference = numpy.sum(absolute_difference, axis=1)
# 4) Create a new column in the movie list with the difference score for each movie
movies_list['difference_score'] = total_difference
# 5) Sort the movie list by difference score, from the least different to most different
sorted_movie_list = movies_list.sort_values('difference_score')
# 6) print the result, showing the 10 most similar movies to movies_id
print('The ten most similar movies are:')
print(sorted_movie_list[['title', 'difference_score']][0:10])
