import tldextract
from tqdm import tqdm
import pandas as pd
import glob
import numpy as np
import re
import argparse

FRAME_COLUMNS = ['age', 'armed', 'attack', 'criminal_record', 'fleeing', 'gender', 'interview', 'legal_language', 'mental_illness', 'official_report', 'race', 'systemic', 'unarmed', 'video']

def to_ordinal(row, prefix):
    d = {}
    for col_name in row.keys():
        if (len(re.findall(prefix, col_name))>0) \
        and (type(row[col_name]) in [float, int, np.float64]):
            if row[col_name]>-1:
                d[col_name] = row[col_name]
    return {k:i+1 for i, (k, v) in enumerate(sorted(d.items(), key=lambda item: item[1]))}

def order_row(row, prefix='found.'):
    orders = np.ones(14)*np.inf
    ordinal = to_ordinal(row, prefix)
    for i, col in enumerate(FRAME_COLUMNS):
        col_name = prefix+col
        if col_name in ordinal:
            orders[i] = ordinal[col_name]
    return orders

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='path to input file', default='data/prepared/shootings/shooting_frames_.csv')
    args = parser.parse_args()
    
    data = []
    for fn in glob.glob('frames-extracted/frames*.csv'):
        data.append(pd.read_csv(fn))
    shootings = pd.concat(data)

    all_ranks = np.stack([order_row(row) for _, row in shootings.iterrows()])

    shootings[['found.'+frame for frame in FRAME_COLUMNS]] = all_ranks

    bias = []
    urls = []
    media_bias = pd.read_csv('resources/mbfc/media-bias-fc-scrape.csv')
    for i, row in media_bias.iterrows():
        val = row['bias_png']
        if 'center' in val:
            bias.append(1)
        elif 'left' in val:
            bias.append(0)
        elif 'right' in val:
            bias.append(2)
        else:
            bias.append(1)
        subdomain, domain, suffix = tldextract.extract(row['url'])
        urls.append(domain)
    media_bias['bias'] = bias
    media_bias['domain'] = urls

    leanings = []
    for i, row in tqdm(shootings.iterrows(), total=len(shootings)):
        bias = media_bias[media_bias['domain']==row['domain']]
        if len(bias)>0:
            leanings.append(bias.iloc[0]['bias'].item())
        else:
            leanings.append(-1)

    shootings['leaning'] = leanings
    shootings.to_csv(args.output)
    
if __name__=='__main__':
    main()