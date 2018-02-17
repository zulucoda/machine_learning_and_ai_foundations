import webbrowser

import numpy
import pandas
import os

# read data from file using pandas
movie_ratings_dt = pandas.read_csv("movie_ratings_data_set.csv")

# convert movie ratings list into matrix using pandas pivot table function
ratings_pivot_table = pandas.pivot_table(movie_ratings_dt, index='user_id', columns='movie_id', aggfunc=numpy.max)

# create html page
matrix_html = ratings_pivot_table.to_html(na_rep='')

with open('review_matrix.html', 'w') as f:
    f.write(matrix_html)

# open html page
full_filename = os.path.abspath('review_matrix.html')
webbrowser.open('file://{}'.format(full_filename))