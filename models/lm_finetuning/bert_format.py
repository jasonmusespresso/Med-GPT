import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
import os
from pathlib import Path
import swifter


ROOT_FOLDER = Path().resolve()
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

df = pd.read_csv(os.path.join(DATA_FOLDER, 'raw', 'AttentionMESH_100k.csv'))

df['abstract'] = df['abstract'].swifter.apply(lambda abstract: '\n'.join(sent_tokenize(abstract)))

with open(os.path.join(DATA_FOLDER, 'interim', 'abstract.txt'), 'w') as fout:
    for abstract in tqdm(df['abstract'].values):
        fout.write('{}\n\n'.format(abstract))