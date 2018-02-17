import numpy
import pandas
import matrix_factorization_utilities

movie_ratings_list = pandas.read_csv('../chapter-4/movie_ratings_data_set.csv')

movie_ratings_pivot_table = pandas.pivot_table(movie_ratings_list, index='user_id',
                                               columns='movie_id', aggfunc=numpy.max)

# use matrix factorisation to find latent features
U,M = matrix_factorization_utilities.low_rank_matrix_factorization(movie_ratings_pivot_table.as_matrix(),
                                                                   num_features=15,
                                                                   regularization_amount=0.1)
# find predicted ratings by multiplying U and M using numpy.matmul
predicted_ratings = numpy.matmul(U, M)

# save ratings to csv
predicted_ratings_data_set = pandas.DataFrame(index=movie_ratings_pivot_table.index,
                                              columns=movie_ratings_pivot_table.columns,
                                              data=predicted_ratings)

predicted_ratings_data_set.to_csv('predicted_ratings.csv')


