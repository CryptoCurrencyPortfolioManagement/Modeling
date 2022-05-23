import numpy as np
import pandas as pd

def standardize_length(pds, length):
    base = np.array([np.nan]*length)
    l = min(length, len(pds.iloc[0]))
    base[:l] = pds.iloc[0][:l]
    pds.iloc[0] = base
    return pds

def generate_texts_per_asset(group, texts_per_day, words_per_text):
    if group.shape[0] > 0:
        if group.shape[0] > texts_per_day:
            group = group.iloc[-texts_per_day:, :]
        else:
            pass
        group = group.apply(lambda x: standardize_length(x, words_per_text), axis= 1)
        temp = pd.Series(index= [group.columns[0]], data= [[item for sublist in group.iloc[:, 0].to_list() for item in sublist]])
        base = np.array([np.nan]*(words_per_text*texts_per_day))
        l = min((words_per_text*texts_per_day), len(temp.iloc[0]))
        base[:l] = temp.iloc[0][:l]
        temp.iloc[0] = base
        return temp
    else:
        return pd.Series(index= [group.columns[0]], data= [[np.nan]*words_per_text*texts_per_day])

def generate_texts_per_day(group, texts_per_day, words_per_text, assets):
    date = group.index[0]
    #group = group.loc[group.iloc[:, 1].apply(lambda x: x.lower() in [asset.lower() for asset in assets])]
    group = group.groupby(group.columns[1]).apply(lambda x: generate_texts_per_asset(x, texts_per_day, words_per_text))
    group = group.transpose()
    group.columns = [x.lower() for x in group.columns]
    for key in list(set(assets)-set(group.columns)):
        group[key] = [[np.nan]*words_per_text*texts_per_day]
    group = group[assets]
    return pd.Series(group.iloc[0])

def generate_nums_per_day(group, assets):
    group = group.reset_index()
    group.iloc[:, -1] = group.iloc[:, -1].str.lower()
    group = group.groupby([group.columns[0], group.columns[-1]]).mean()
    group = group.reset_index()
    group = group.pivot(index=group.columns[0], columns=group.columns[1])
    group.index = pd.to_datetime(group.index).date
    for column in group.columns.get_level_values(0).unique():
        for key in list(set(assets)-set(group[column].columns)):
            group[column][key] = [np.nan]
    group.columns = group.columns.map('|'.join).str.strip('|')
    return group