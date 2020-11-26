import os
import pandas as pd

vidPath = '/home/clionaodoherty/foundcog_stimuli/trimmed'

df = pd.DataFrame(index=os.listdir(vidPath), columns=['filesize'])

for f in os.listdir(vidPath):
    df.loc[f] = os.stat(os.path.join(vidPath,f)).st_size

df.to_csv('./complexity.csv')