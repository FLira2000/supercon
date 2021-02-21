import pandas as pd
from makeup import makeup
from prob_ratios import prob_ratios

def chemical_extraction(element_dict):
    subset_table = pd.read_csv('./../superconduct/subset_element_data_fixed.csv')
    
    subset_selected = []
    for chem in element_dict.keys():
        subset_selected.append(subset_table.loc[subset_table['Element'] == chem].to_dict())
    
    tmp = 0
    for row in subset_selected:
        tmp = list(row['FullNameElement'].keys())[0]
        for key in row.keys():
            row[key] = row[key][tmp]
        
    subset_selected_df = pd.DataFrame(subset_selected)

    element_ratios = prob_ratios(element_dict)

    for chem in list(element_ratios.items()):
        subset_selected_df.loc[subset_selected_df.Element == chem[0], 'ProbRatios'] = chem[1]

    return subset_selected_df

tmp = chemical_extraction(makeup('C6H12O6'))
print(tmp.to_dict())
