#%%
import pandas as pd
import numpy as np


enrichment = False

if enrichment:
    enrichment_df = pd.read_csv("maml_results.txt", sep="\t")
    enrichment_df.set_index('original_name',inplace=True)
    repnames2 = enrichment_df.to_dict()["enrichment_term"]

filepath = "Network_.Sparsity_99.91.TrainAcc_99.292.TestAcc_97.466.csv"
#filepath = "../experiments/Naive_10Sig/results/Network_.Sparsity_99.19.TrainAcc_100.0.TestAcc_98.808.csv"
#%% select a file 
df = pd.read_csv(filepath)
df = df[~df["source_id"].str.contains("input")]

a = []
temp_list = list(df["source_id"])

for elem in temp_list:
    if elem[0].islower():
        a.append(elem.split("_")[0])
    else:
        a.append(elem)
df["source_id"] = a

df_source = df.source_id
df_target = df.target_id

df_nodes = np.unique(list(df_source) + list(df_target))

repnames = dict({'Root_0':'BRCA','Root_1':'COAD','Root_2':'LUAD','Root_3':'OV','Root_4':'THCA'})

# replace names if you can, e.g. by enrichment analysis of the connected terms 
"""repnames2 = dict({'GO-01_6':'Thyroid Gland Development (1) (GO:0030878)',
                  'GO-01_23':'Intracellular sequestering of iron ion (2) (GO:0006880)',
                  'GO-01_16':'Thyroid gland development (2) (GO:0030878)',
                  'GO-01_0':'Intracellular sequestering of iron ion (1) (GO:0006880)',
                  'GO-01_1':'Positive regulation of autophagosome assembly (GO:2000786)'})
"""

# %% create connection matrix
df_connection = pd.DataFrame(0,columns=df_nodes,index=df_nodes)
for s in df_nodes:
    s_id = list(np.where(df_nodes == s))[0]
    print("checking {0}".format(s))
    for t in df_nodes:
        t_id = list(np.where(df_nodes == t))[0]
        if df[df.source_id == s].shape[0]>0:
            if t in list(df[df.source_id == s].target_id):
                edges = df[(df.source_id == s) & (df.target_id==t)].edge_value
                df_connection.iloc[s_id,t_id]=edges
          
        
#%% store to data
nanmin = np.nanmin(df_connection.replace(0,np.nan))

enrich_dict = {}
for col in df_connection:
    enrich_dict[col] = list(df_connection[df_connection[col] != 0].index)

with open("repnames.txt", "w") as repfile:
    for original_name, gene_set in enrich_dict.items():
        if "Root" in original_name:
            continue
        repfile.write(original_name + "\n")
        for gene in gene_set:
            repfile.write(gene.split("_")[0] + "\n")
        repfile.write("\n")




df_connection_export = np.round(df_connection/nanmin,0).astype(int)
df_connection_export = df_connection_export.rename(index=repnames)

if enrichment:
    df_connection_export = df_connection_export.rename(index=repnames2)
df_connection_export.to_csv("connection_map.txt",header=None,sep='\t')



