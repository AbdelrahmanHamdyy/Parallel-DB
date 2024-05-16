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
#define BLOCK_SIZE 256

__global__ void parallelLinearSearch(int arr[], int size, int *output, int target) {
    *output = -1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && arr[index] == target) {
        *output = index;
    }
}

void linearSearchWrapper(int arr[], int size, int target) {
    int *d_arr, *d_output;
    int output;
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
                char column[50];
                printf("SELECT * FROM table1 WHERE ");
                scanf("%s", column);
                int target;
                printf("EQUAL TO ");
                scanf("%d", &target);
                int data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                int size = readCSV("table_1.csv", data_tokens, data, column, &columnIndex);
                
                // Parallel Linear Search
                linearSearchWrapper(data, size, target);

                free(data_tokens);
                break;
            }
            case 2: {
                char column[50];
                printf("SELECT * FROM table1 ORDER BY ");
                scanf("%s", column);
                int data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                int size = readCSV("table_1.csv", data_tokens, data, column, &columnIndex);
                
                // Parallel Merge Sort

                printf("Sorting complete.\n");
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
                printf("SELECT * FROM table1 INNER JOIN table2 ON table1.");
                char column1[50], column2[50];
                scanf("%s", column1);
                printf("EQUAL table2.");
                scanf("%s", column2);
                int data1[MAX_VALUES], data2[MAX_VALUES], columnIndex1, columnIndex2;
                char** data_tokens1 = (char**)malloc(MAX_VALUES * sizeof(char*));
                char** data_tokens2 = (char**)malloc(MAX_VALUES * sizeof(char*));
                int size1 = readCSV("table_2.csv", data_tokens1, data1, column1, &columnIndex1);
                int size2 = readCSV("table_3.csv", data_tokens2, data2, column2, &columnIndex2);

                // Perform inner join operation
                printf("Inner join result:\n");
                
                // Parallel Inner Join

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