import sys
sys.path.insert(0, '../chapter-5')

import numpy
import pandas
import pickle
import matrix_factorization_utilities


users_movie_ratings_list = pandas.read_csv('../chapter-4/movie_ratings_data_set.csv')


users_movie_ratings_pivot_table = pandas.pivot_table(users_movie_ratings_list, index='user_id', columns='movie_id',
                                                     aggfunc=numpy.max)


U,M = matrix_factorization_utilities.low_rank_matrix_factorization(users_movie_ratings_pivot_table.as_matrix(),
                                                                   num_features=15,
                                                                   regularization_amount=0.1)

predicted_ratings = numpy.matmul(U, M)

# save features and predicted ratings to files for later use
pickle.dump(U, open('user_features.dat', 'wb'))
pickle.dump(M, open('product_features.dat', 'wb'))
pickle.dump(predicted_ratings, open('predicted_ratings.dat', 'wb'))
