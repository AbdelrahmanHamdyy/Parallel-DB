// We will be using CUDA to parallelize the following DB operations
// 1. Linear Search
// 2. Merge Sort
// 3. Inner Join

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define MAX_VALUES 100
#define MAX_MATCHES (MAX_VALUES * MAX_VALUES)
#define BLOCK_SIZE 256

// Function to split a string by delimiter
char** split(const char* s, char delimiter, int* num_tokens) {
    char** tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
    if (tokens == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    char* token = strtok((char*)s, ",");
    *num_tokens = 0;
    while (token != NULL) {
        tokens[*num_tokens] = strdup(token);
        if (tokens[*num_tokens] == NULL) {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        (*num_tokens)++;
        token = strtok(NULL, ",");
    }
    return tokens;
}

// Function to read a CSV file
int readCSV(const char *filename, char **data, int arr[], char *column, int *columnIndex) {
    FILE* file = fopen(filename, "r");
    int size = 0;
    char line[1024];
    bool firstLine = true;
    int num_tokens;
    while (fgets(line, sizeof(line), file)) {
        if (firstLine) {
            char** tokens = split(line, ',', &num_tokens);
            for (int i = 0; i < num_tokens; i++) {
                if (strncmp(column, tokens[i], strlen(column)) == 0) {
                    *columnIndex = i;
                    break;
                }
            }
            firstLine = false;
            continue;
        }
        data[size] = strdup(line);
        char** tokens = split(line, ',', &num_tokens);
        arr[size++] = atoi(tokens[*columnIndex]);
        free(tokens);
    }
    fclose(file);
    return size;
}

// Function to check if an array is sorted
bool isSorted(int* arr, int size)  {
    for (int i = 1; i < size; ++i) {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

__global__ void parallelLinearSearch(int arr[], int size, int *output, int target) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && arr[index] == target) {
        *output = index;
    }
}

__device__ void merge(int arr[], int sorted[], int start, int mid, int end) {
    int k = start, i = start, j = mid;
    while (i < mid || j < end)
    {
        if (j == end) sorted[k] = arr[i++];
        else if (i == mid) sorted[k] = arr[j++];
        else if (arr[i] < arr[j]) sorted[k] = arr[i++];
        else sorted[k] = arr[j++];
        k++;
    }
}

__global__ void parallelMergeSort(int arr[], int sorted[], int size, int chunkSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * chunkSize;
    if (start >= size) return;
    int mid = min(start + chunkSize / 2, size);
    int end = min(start + chunkSize, size);
    merge(arr, sorted, start, mid, end);
}

__global__ void parallelInnerJoin(int arr1[], int arr2[], int *matchedIndices, int *numMatches, int size1, int size2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size1) {
        for (int i = 0; i < size2; i++) {
            if (arr1[index] == arr2[i]) {
                int matchIndex = atomicAdd(numMatches, 1);
                if (matchIndex < MAX_MATCHES) {
                    matchedIndices[matchIndex * 2] = index;
                    matchedIndices[(matchIndex * 2) + 1] = i;
                }
            }
        }
    }
}

void linearSearchWrapper(int arr[], int size, int target) {
    int *d_arr, *d_output;
    int output = -1;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &output, sizeof(int), cudaMemcpyHostToDevice);
    parallelLinearSearch<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_arr, size, d_output, target);
    cudaDeviceSynchronize();
    cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    if (output == -1) {
        printf("Element not found.\n");
    } else {
        printf("Element found at index %d.\n", output);
    }
    cudaFree(d_arr);
    cudaFree(d_output);
}

void mergeSortWrapper(int arr[], int size) {
    int *d_arr, *d_sorted;
    cudaMalloc((void **)&d_arr, size * sizeof(int));
    cudaMalloc((void **)&d_sorted, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    int numberOfBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 2; i < size * 2; i *= 2) {
        parallelMergeSort<<<numberOfBlocks, BLOCK_SIZE>>>(d_arr, d_sorted, size, i);
        cudaDeviceSynchronize();
        // Swap
        int *temp = d_arr;
        d_arr = d_sorted;
        d_sorted = temp;
    }
    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_sorted);
}

void innerJoinWrapper(int arr1[], int arr2[], char **data1, char **data2, int size1, int size2) {
    int *d_arr1, *d_arr2;
    int *numMatches = (int*)malloc(sizeof(int));
    *numMatches = 0;
    int *matchedIndices = (int*)malloc(MAX_MATCHES * 2 * sizeof(int));
    int *d_matchedIndices, *d_numMatches;

    cudaMalloc(&d_arr1, size1 * sizeof(int));
    cudaMalloc(&d_arr2, size2 * sizeof(int));
    cudaMalloc(&d_matchedIndices, MAX_MATCHES * 2 * sizeof(int));
    cudaMalloc(&d_numMatches, sizeof(int));

    cudaMemcpy(d_arr1, arr1, size1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, size2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numMatches, numMatches, sizeof(int), cudaMemcpyHostToDevice);

    int numberOfBlocks = (size1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parallelInnerJoin<<<numberOfBlocks, BLOCK_SIZE>>>(d_arr1, d_arr2, d_matchedIndices, d_numMatches, size1, size2);
    cudaDeviceSynchronize();

    cudaMemcpy(numMatches, d_numMatches, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matchedIndices, d_matchedIndices, *numMatches * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Inner Join result:\n");
    for (int i = 0; i < *numMatches; i++) {
        int index1 = matchedIndices[i * 2];
        int index2 = matchedIndices[(i * 2) + 1];
        printf("%s%s\n", data1[index1], data2[index2]);
    }

    free(numMatches);
    free(matchedIndices);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_matchedIndices);
    cudaFree(d_numMatches);
}

int main() {
    int choice;
    printf("Choose an operation:\n");
    printf("1. Search\n");
    printf("2. Sort\n");
    printf("3. Join\n");

    while (1) {
        printf("Choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1: {
                char column[50], table[50];
                printf("SELECT * FROM ");
                scanf("%s", table);
                printf("WHERE ");
                scanf("%s", column);
                int target;
                printf("EQUAL TO ");
                scanf("%d", &target);
                int data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                int size = readCSV(table, data_tokens, data, column, &columnIndex);
                
                // Parallel Linear Search
                linearSearchWrapper(data, size, target);

                free(data_tokens);
                break;
            }
            case 2: {
                char column[50], table[50];
                printf("SELECT * FROM ");
                scanf("%s", table);
                printf("ORDER BY ");
                scanf("%s", column);
                int data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                int size = readCSV(table, data_tokens, data, column, &columnIndex);
                
                // Parallel Merge Sort
                mergeSortWrapper(data, size);

                printf("Sorted Data:\n");
                bool *visited = (bool*)malloc(size * sizeof(bool));
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        int num_tokens;
                        char *tokens = strdup(data_tokens[j]);
                        char** db_info = split(tokens, ',', &num_tokens);
                        if (atoi(db_info[columnIndex]) == data[i] && !visited[j]) {
                            visited[j] = true;
                            printf("%s", data_tokens[j]);
                            break;
                        }
                    }
                }
                free(data_tokens);
                break;
            }
            case 3: {
                char table1[50], table2[50];
                printf("SELECT * FROM ");
                scanf("%s", table1);
                printf("INNER JOIN ");
                scanf("%s", table2);
                printf("ON %s.", table1);
                char column1[50], column2[50];
                scanf("%s", column1);
                printf("EQUAL %s.", table2);
                scanf("%s", column2);
                int data1[MAX_VALUES], data2[MAX_VALUES], columnIndex1, columnIndex2;
                char** data_tokens1 = (char**)malloc(MAX_VALUES * sizeof(char*));
                char** data_tokens2 = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table1, ".csv");
                int size1 = readCSV(table1, data_tokens1, data1, column1, &columnIndex1);
                strcat(table2, ".csv");
                int size2 = readCSV(table2, data_tokens2, data2, column2, &columnIndex2);

                // Parallel Inner Join
                innerJoinWrapper(data1, data2, data_tokens1, data_tokens2, size1, size2);

                free(data_tokens1);
                free(data_tokens2);
                break;
            }
            default:
                printf("Invalid choice.\n");
                break;
        }
    }

    return EXIT_SUCCESS;
}