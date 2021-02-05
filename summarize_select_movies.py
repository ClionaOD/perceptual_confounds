import pandas as pd
import ast
from collections import Counter
from matplotlib import pyplot as plt

movies = []
movies8 = []
with open('model_efficiency_select_movies_set2.csv','r') as f:
    df = pd.read_csv(f)
    
    # How often does each of the movies fall in the last eight?
    for movies_str in df['movies']:
        rowaslist = ast.literal_eval(movies_str)
        movies.extend(rowaslist)
        movies8.append(','.join(sorted(rowaslist)))
    
    # Histogram and order it
    dist= Counter(movies)
    dist_order = [i[0] for i in sorted(dist.items(), key=lambda x:x[1])]

    print(dist_order)
    fig, ax = plt.subplots()
    plt.bar(dist_order, [dist[x] for x in dist_order])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('movie_efficiency_select_movies_bar.jpg')


    # What is the most common combination of eight movies?
    dist= Counter(movies8)
    dist_order = [i[0] for i in sorted(dist.items(), key=lambda x:x[1])]
    print('From least to most popular')
    for item in dist_order:
        print(f'{dist[item]} times: {item}')
