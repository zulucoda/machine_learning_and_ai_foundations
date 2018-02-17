import pandas
import numpy

movie_ratings_list = pandas.read_csv('movie_ratings_data_set.csv')

movie_ratings_pivot_table = pandas.pivot_table(movie_ratings_list, index='user_id', columns='movie_id', aggfunc=numpy.max)

movie_ratings_pivot_table.to_csv('review_matrix_generated.csv', na_rep='')
