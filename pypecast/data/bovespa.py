import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from pypecast.utils.path_finder import absoluteFilePaths 

def read_hft_data(path):
    headers = ['Data','Simb','NumNeg','Preco','Quant',
        'Horario','IndAnulacao','DataOferCpa','SeqOferCPA','GenIDCpa',
       'CondOferCpa','DataOferVda','SeqOferVda', 'GenIDVda','CondOferVda',
       'IndDireto','CorrCpaa','CorrVda']
    data = pd.read_csv(path, delimiter='"\s+|;|"', names=headers, header=1)
    frame = preprocess(data)
    return frame

def merge_datasets(input_path, output_path, n_databases = 200):
    headers = ['Unnamed: 0','Data','Simb','Hora','Min','Med','Var']
    all_paths = absoluteFilePaths(input_path)
    #frame = pd.DataFrame(columns=headers)
    list_ = []
    i = 0
    print('Merging preprocesed datasets...')
    for file_ in tqdm(all_paths):
        data = pd.read_csv(file_)
        list_.append(data.values)
        if i>n_databases:
            break
        i += 1
    frame = pd.DataFrame(np.concatenate(list_),columns=headers)
    try:
        frame = frame.drop('Unnamed: 0', axis =1)
    except:
        pass
    frame.to_csv(os.path.join(output_path, 'merged_'+str(i)+'_days.csv'))
    print('{} dataset(s) merged succesfully!'.format(i))

def prep_and_save(in_path, out_path):
    all_paths = absoluteFilePaths(in_path)
    for file_ in all_paths:
        data = read_hft_data(file_)
        val_dia = data.Data.iloc[0]
        try:
            data = data.drop(['Unnamed: 0'], axis=1)
        except:
            pass
        filename = os.path.join(out_path,val_dia+'.csv')
        data.to_csv(filename)

def remove_cols(data, removable = None):
    #Removing unecessary collumns and metadata rows
    if removable is None:
        removable = ['CorrCpaa', 'CorrVda','DataOferVda', 'DataOferCpa', 'SeqOferCPA','SeqOferVda',
                        'GenIDCpa','GenIDVda','IndAnulacao', 'CondOferCpa', 'CondOferVda', 'IndDireto', 'NumNeg']
    data= data.drop(removable, axis=1)
    data= data.drop(data.shape[0]-1)
    #Symbol space removal
    data.Simb = data.Simb.apply(lambda x: str(x).strip())
    return data
    
def keep_subset(data, keep):
    #Keeps the highest liquidity stocks from the dataset
    data = data[data['Simb'].isin(keep)]
    return data

def prep_horario(data, keep, gran = 1, ini_crop = 30, end_crop = 30):
    assert gran < 60
    # Processing the hours, mins and seconds
    data.Horario = data.Horario.apply(lambda x: x.split(':'))
    data[['Hora','Min','Seg']] = pd.DataFrame(data.Horario.values.tolist(), index= data.index)
    data.Hora = data.Hora.apply(lambda x: int(x) - 10)
    data = data.drop(['Horario', 'Seg'], axis = 1)
    data.Min = data.Min.apply(lambda x: int(x))
    # Removing the initial and final 20 minutes
    init, end = 29, 30
    data = data[~((data.Hora==0) & (data.Min<29))]
    data = data[~((data.Hora==7) & (data.Min>30))]
    # Constant data
    headers = ['Data', 'Simb','Hora', 'Min', 'Med', 'Var']
    dia = data.Data.iloc[0]
    #Building a new dataframe
    frame = pd.DataFrame(columns=headers)
    list_ = []
    for ticker in tqdm(keep):
        for hour in range(8):
            if hour == 0:
                init = ini_crop
                end = 60
            elif hour == 7:
                init = 0
                end = end_crop
            else:
                init = 0
                end = 60
            tmp = data[(data.Simb == ticker) & (data.Hora == hour)]
            for k in range(init,end,gran):
                try:
                    #Mean of k, k + gran interval
                    mean = tmp.Preco.groupby((tmp.Min >= k) & (tmp.Min < k+gran)).mean()[True]
                    var = tmp.Preco.groupby((tmp.Min >= k) & (tmp.Min < k+gran)).var()[True]

                    to_list = pd.DataFrame(data = [[dia, ticker, hour, k, mean, var]],
                                         columns=headers)
                except:
                    #If there are no values in an interval, it keeps the last one
                    to_list = pd.DataFrame(data = [[dia, ticker, hour, k, mean, var]],
                                         columns=headers)                    
                list_.append(to_list)
    frame = pd.concat(list_)
    return frame

def preprocess(data,  keep_list = None,granularity=1):
    data = remove_cols(data)
    if keep_list is None:
        keep_list = ['PETR4', 'BRFS3', 'BBAS3', 'BBDC4', 'ITSA4']
    data = keep_subset(data, keep_list)
    data = prep_horario(data, keep_list)
    return data