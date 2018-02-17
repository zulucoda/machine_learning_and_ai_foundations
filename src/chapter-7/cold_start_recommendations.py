import pickle

import pandas

means = pickle.load(open('means.dat', 'rb'))

movies_list = pandas.read_csv('../chapter-4/movies.csv', index_col='movie_id')

user_ratings = means

print('Movies we will recommend:')

movies_list['rating'] = user_ratings
movies_list = movies_list.sort_values(by=['rating'], ascending=False)

print(movies_list[['title','genre','rating']].head(10))