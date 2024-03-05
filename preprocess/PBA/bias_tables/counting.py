import pandas as pd

file = pd.read_csv("/mnt/sdc/glee623/projects/Debias/gene-cft-2/preprocess/PBA/bias_tables/cmnist/0.6_final/bias_table.csv")

dict_list = {"align 1 0" : 0,"align 0 1" : 0, "conflict 1 0" : 0, "conflict 0 1" : 0}
for i in range(len(file)):
    file_path, bias, bias_conflict = file.iloc[i][:3]
    decision = file_path.split("/")[5]

    if decision == "align":
        if bias == 1 and bias_conflict == 0:
            dict_list["align 1 0"] += 1
        if bias == 0 and bias_conflict == 1:
            dict_list["align 0 1"] += 1
    elif decision == "conflict":
        if bias == 1 and bias_conflict == 0:
            dict_list["conflict 1 0"] += 1
        if bias == 0 and bias_conflict == 1:
            dict_list["conflict 0 1"] += 1
    

print(dict_list)

