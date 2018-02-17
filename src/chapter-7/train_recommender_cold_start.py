import sys
sys.path.insert(0, '../chapter-5')

import pandas
import numpy
import pickle

import matrix_factorization_utilities


users_movie_ratings_list = pandas.read_csv('../chapter-4/movie_ratings_data_set.csv')


users_movie_ratings_pivot_table = pandas.pivot_table(users_movie_ratings_list, index='user_id', columns='movie_id',
                                                     aggfunc=numpy.max)

# normalise ratings around their mean
normalise_ratings, means = matrix_factorization_utilities.normalize_ratings(users_movie_ratings_pivot_table.as_matrix())

U,M = matrix_factorization_utilities.low_rank_matrix_factorization(normalise_ratings,
                                                                   num_features=11,
                                                                   regularization_amount=1.1)

predicted_ratings = numpy.matmul(U, M)

predicted_ratings = predicted_ratings + means

pickle.dump(U, open('user_features.dat', 'wb'))
pickle.dump(M, open('product_features.dat', 'wb'))
pickle.dump(predicted_ratings, open('predicted_ratings.dat', 'wb'))
pickle.dump(means, open('means.dat', 'wb'))