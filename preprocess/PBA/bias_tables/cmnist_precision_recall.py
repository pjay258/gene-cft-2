import pandas as pd

csv_file = pd.read_csv("/mnt/sdc/glee623/projects/Debias/gene-cft-2/preprocess/PBA/bias_tables/cmnist/0.6_final/bias_table.csv")

file_path = csv_file["file_path"]
bias_align = csv_file["bias"].tolist()
bias_conflict = csv_file["bias_conflict"].tolist()


results = []
for i in range(len(file_path)):
    dir = file_path[i].split('/')[6] # file dir이 align인지 con인지
    b_align = bias_align[i] 
    b_conflict = bias_conflict[i]
    
    results.append([dir, b_align, b_conflict])

c_c = 0
a_c = 0
c = 0

for i in range(len(results)):
    if results[i][0] == 'conflict':
        c += 1
    if results[i][0] == 'conflict' and results[i][2] == 1:
        c_c += 1
    if results[i][0] == 'align' and results[i][2] == 1: 
        a_c += 1

print(c)
print(c_c)
print(a_c)

precision = c_c / (a_c + c_c)
recall = c_c / c
print("precision",precision)
print("recall", recall)



