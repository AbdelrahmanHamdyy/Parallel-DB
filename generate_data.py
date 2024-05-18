import csv
import random

SIZE = 10000
MAX_VALUE = 10000

def generate():
    '''
    data.csv will be used for searching and sorting operations
    data1.csv and data2.csv will be used for join operation
    '''
    with open('table_1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
    with open('table_2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
    with open('table_3.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value', 'symbol'])
        for i in range(SIZE):
            writer.writerow([i, random.randint(0, MAX_VALUE), chr(random.randint(65, 90))])
            
generate()