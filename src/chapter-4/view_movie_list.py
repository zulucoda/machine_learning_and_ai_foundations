import os
import webbrowser

import pandas

movie_list_table = pandas.read_csv("movies.csv", index_col="movie_id")

html = movie_list_table.to_html()

with open("movie_list.html", "w") as f:
    f.write(html)

full_filename = os.path.abspath("movie_list.html")
webbrowser.open("file://{}".format(full_filename))