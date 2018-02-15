import webbrowser

import os

import pandas as pandas

data_table = pandas.read_csv("movie_ratings_data_set.csv")

html = data_table[0:100].to_html()

with open("data.html", "w") as f:
    f.write(html)

full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))