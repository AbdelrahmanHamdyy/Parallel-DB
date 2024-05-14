import csv
import random

SIZE = 10
MAX_VALUE = 20

def generate():
    '''
    data.csv will be used for searching and sorting operations
    data1.csv and data2.csv will be used for join operation
    '''
    with open('table1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
    with open('table2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
    with open('table3.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
generate()