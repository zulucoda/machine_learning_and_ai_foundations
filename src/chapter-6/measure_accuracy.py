import sys
sys.path.insert(0, '../chapter-5')

import pandas
import numpy
import matrix_factorization_utilities

users_movie_ratings_training = pandas.read_csv('movie_ratings_data_set_training.csv')
users_movie_ratings_testing = pandas.read_csv('movie_ratings_data_set_testing.csv')

users_movie_ratings_training_pivot_table = pandas.pivot_table(users_movie_ratings_training, index='user_id', columns='movie_id',
                                                     aggfunc=numpy.max)
users_movie_ratings_testing_pivot_table = pandas.pivot_table(users_movie_ratings_testing, index='user_id', columns='movie_id',
                                                     aggfunc=numpy.max)


U,M = matrix_factorization_utilities.low_rank_matrix_factorization(users_movie_ratings_training_pivot_table.as_matrix(),
                                                                   num_features=11,
                                                                   regularization_amount=1.1)

predicted_ratings = numpy.matmul(U, M)


# Measure RMSE
rmse_training = matrix_factorization_utilities.RMSE(users_movie_ratings_training_pivot_table.as_matrix(),
                                                    predicted_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(users_movie_ratings_testing_pivot_table.as_matrix(),
                                                   predicted_ratings)

print('Training RMSE: {}'.format(rmse_training))
print('Testing RMSE: {}'.format(rmse_testing))

