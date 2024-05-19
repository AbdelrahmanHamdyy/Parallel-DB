# Parallel-DB
Parallelized DB Operations using Cuda.

# Application README

## Overview
This application is designed for data generation, GPU-accelerated search, join operations, and bitonic sorting. Follow the instructions below to set up and run the application.

## Prerequisites
- CUDA Toolkit installed
- Python installed
- Jupyter Notebook (optional but recommended for interactive use)

## Setup Instructions

### 1. Configure Data Size
Before running the application, you need to specify the data size. This requires changes in three files:

1. Open `generate_data.py` and set the `SIZE` variable:
    ```python
    SIZE = <desired_data_size>
    ```
2. Open `gpu.cu` and set the `MAX_VALUES` macro:
    ```cpp
    #define MAX_VALUES <desired_data_size>
    ```
3. Open `sort.cu` and set the `ORIGINAL_NUMBER` variable:
    ```cpp
    int ORIGINAL_NUMBER=100000; <desired_data_size>
    ```

### 2. Generate Data
Run the data generation script:
```sh
!python ./generate_data.py
```
### 3. Install CUDA Jupyter Extension
To enable CUDA execution in Jupyter notebooks, install and load the `nvcc4jupyter` extension:

```sh
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
## Running the Application

### Search and Join Operations

#### Compile the GPU code:
```sh
!nvcc -g -G gpu.cu -o gpu
!./gpu
```

Follow the prompts in the console:

### For a search operation:

```sql
SELECT * FROM _____
```
Replace _____ with the table name (e.g., table_1, table_2, or table_3).

```sql
WHERE ______
```
Replace ______ with the column name (e.g., value).

```sql
EQUAL TO _______
```
Specify the value you want to search for.

### For a join operation:

```sql
SELECT * FROM _______
```
Replace _______ with the first table name.
```sql
INNER JOIN ________
```
Replace ________ with the second table name.
```sql
ON tableOneName. _______
```
Replace _______ with the column name from the first table.
```sql
EQUAL tableTwoName. ________
```
Replace ________ with the column name from the second table.

### for a merge sort
Follow the prompt in the console:
```sql
SELECT * FROM _______ ORDER BY________
```
Replace _______ with the table name and ________ with the column name to sort by.


## Bitonic Sort

#### Compile the sorting code:
```sh
!nvcc -g -G sort.cu -o sort
!./sort
```
Follow the prompt in the console:
```sql
SELECT * FROM _______ ORDER BY________
```
Replace _______ with the table name and ________ with the column name to sort by.
