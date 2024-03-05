import pandas as pd

csv_file = pd.read_csv("/mnt/sdc/glee623/projects/Debias/gene-cft-2/preprocess/PBA/bias_tables/bffhq/0.1_final/bias_table.csv")

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


'''
0.6
---
96
42
81
precision 0.34146341463414637
recall 0.4375

0.5
---
96
40
70
precision 0.36363636363636365
recall 0.4166666666666667

0.1
---
96
44
50
precision 0.46808510638297873
recall 0.4583333333333333

0.05
---
96
40
49
precision 0.449438202247191
recall 0.4166666666666667


0.001
---
96
30
23
precision 0.5660377358490566
recall 0.312

0.0001
---
96
25
14
precision 0.6410256410256411
recall 0.2604166666666667
'''