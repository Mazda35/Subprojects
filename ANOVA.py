
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def anovaFN(IC50='LUAD_IC.csv',features='LUAD_simple_MOBEM.rdata.tsv'):
    # IC50 is based on cancerrxgene.org data
    # features are BEM files from ayestaran data
    # Columns and Names are absed on these two data,
    # if you intend to use another data, send me format to modify this code.

    ##### IMPORTANT  =>  We have

    ic50 = pd.read_csv(IC50)
    ic=ic50.pivot(index='Cosmic sample Id',columns='Drug Id',values='IC50')
    ic.index.name = None
    ic.columns.name = None

    drugs=ic.columns.tolist()

    bem=pd.read_csv(features,sep='\t')
    bem = bem.set_index('Unnamed: 0')
    bem.index.name = None

    bemT = bem.transpose()
    bemT.index = bemT.index.astype('int64')

    mutations=bemT.columns.tolist()

    bem_drug = bemT.merge(ic, left_index=True, right_index=True)

    data=[]

    for drug_id in drugs:
        drug_name=ic50.loc[ic50['Drug Id'] == drug_id, 'Drug name'].unique().tolist()
        for mut in mutations:
            y = bem_drug[drug_id].dropna()
            ind = y.index
            if bemT.loc[ind, mut].sum() > 2:
                masked_features = bemT.loc[ind, :]
                positive_features = masked_features.values.sum()
                negative_features = len(masked_features) - positive_features
                positive = masked_features[masked_features == 1]

                n_pos = bemT.loc[ind, mut].sum()
                n_neg = len(bemT.loc[ind, mut]) - bemT.loc[ind, mut].sum()

                pos_mean = bem_drug.loc[bemT[mut] == 1, drug_id].mean()
                neg_mean = bem_drug.loc[bemT[mut] == 0, drug_id].mean()

                pos_var = bem_drug.loc[bemT[mut] == 1, drug_id].var()
                neg_var = bem_drug.loc[bemT[mut] == 0, drug_id].var()

                pos_std = bem_drug.loc[bemT[mut] == 1, drug_id].std()
                neg_std = bem_drug.loc[bemT[mut] == 0, drug_id].std()

                md = np.abs(pos_mean - neg_mean)
                cv = (((n_pos - 1) * pos_var) + ((n_neg - 1) * neg_var)) / (n_pos + n_neg - 2)
                effect_size=md/np.sqrt(cv)

                test=ttest_ind(bem_drug.loc[bemT[mut]==1,drug_id],
                               bem_drug.loc[bemT[mut]==0,drug_id])
                pval=test.pvalue

                data.append([drug_name[0],drug_id,mut,n_pos,n_neg,
                             round(pos_mean,8),round(neg_mean,8),round(md,8),
                             round(effect_size,8),round(pos_std,8),round(neg_std,8),pval])

    anova=pd.DataFrame(data,columns=['Drug_Name','Drug_ID','Feature_Name',
                                'pos_n_feature','neg_n_feature',
                                'pos_ic50_mean','neg_ic50_mean',
                                'delta_ic50_mean','effect_size',
                                'pos_ic50_variance','neg_ic50_variance',
                                'feature_ic50_p_value'])

    return anova

anova=anovaFN()
anova.to_csv('test.csv')

