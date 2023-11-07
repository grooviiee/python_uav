import csv

f = open('example.csv', 'r')
rdr = csv.reader(f)

for line in rdr:
    print(line)
    
f.close()