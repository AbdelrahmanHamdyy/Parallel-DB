// We will be using CUDA to parallelize the following DB operations
// 1. Linear Search
// 2. Merge Sort
// 3. Inner Join

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

typedef long long ll;

#define MAX_VALUES 1000
#define MAX_MATCHES ((ll)MAX_VALUES * MAX_VALUES)
#define BLOCK_SIZE 256

// Function to split a string by delimiter
char** split(const char* s, char delimiter, ll* num_tokens) {
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
ll readCSV(const char *filename, char **data, ll arr[], char *column, ll *columnIndex) {
    FILE* file = fopen(filename, "r");
    ll size = 0;
    char line[1024];
    bool firstLine = true;
    ll num_tokens;
    while (fgets(line, sizeof(line), file)) {
        if (firstLine) {
            char** tokens = split(line, ',', &num_tokens);
            for (ll i = 0; i < num_tokens; i++) {
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
bool isSorted(ll* arr, ll size)  {
    for (ll i = 1; i < size; ++i) {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

__global__ void parallelLinearSearch(ll arr[], ll size, ll *matchedIndices, int *numMatches, ll target) {
    __shared__ ll sharedArr[BLOCK_SIZE];
    ll index = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < size) {
        sharedArr[threadIdx.x] = arr[index];
    }
    __syncthreads();
    if (index < size) {
        if (sharedArr[threadIdx.x] == target) {
            ll matchIndex = (ll)atomicAdd(numMatches, 1);
            if (matchIndex < MAX_VALUES) {
                matchedIndices[matchIndex] = index;
            }
        }
    }
}

__device__ void merge(ll arr[], ll sorted[], ll start, ll mid, ll end) {
    ll k = start, i = start, j = mid;
    while (i < mid || j < end)
    {
        if (j == end) sorted[k] = arr[i++];
        else if (i == mid) sorted[k] = arr[j++];
        else if (arr[i] < arr[j]) sorted[k] = arr[i++];
        else sorted[k] = arr[j++];
        k++;
    }
}

__global__ void parallelMergeSort(ll arr[], ll sorted[], ll size, ll chunkSize) {
    ll tid = threadIdx.x + blockIdx.x * blockDim.x;
    ll start = tid * chunkSize;
    if (start >= size) return;
    ll mid = min(start + chunkSize / 2, size);
    ll end = min(start + chunkSize, size);
    merge(arr, sorted, start, mid, end);
}

__global__ void parallelInnerJoin(ll arr1[], ll arr2[], ll *matchedIndices, int *numMatches, ll size1, ll size2) {
    __shared__ ll sharedArr2[BLOCK_SIZE];
    ll index = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < size2) {
        sharedArr2[threadIdx.x] = arr2[index];
    }
    __syncthreads();
    if (index < size1) {
        for (ll i = 0; i < size2; i++) {
            if (arr1[index] == sharedArr2[i]) {
                ll matchIndex = (ll)atomicAdd(numMatches, 1);
                if (matchIndex < MAX_MATCHES) {
                    matchedIndices[matchIndex * 2] = index;
                    matchedIndices[(matchIndex * 2) + 1] = i;
                }
            }
        }
    }
}

void linearSearchWrapper(ll arr[], char **data, ll size, ll target) {
    ll *d_arr;
    int *numMatches = (int*)malloc(sizeof(int));
    *numMatches = 0;
    ll *matchedIndices = (ll*)malloc(MAX_VALUES * sizeof(ll));
    ll *d_matchedIndices;
    int *d_numMatches;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_arr, size * sizeof(ll));
    cudaMalloc(&d_matchedIndices, MAX_VALUES * sizeof(ll));
    cudaMalloc(&d_numMatches, sizeof(int));

    cudaMemcpyAsync(d_arr, arr, size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_numMatches, numMatches, sizeof(int), cudaMemcpyHostToDevice, stream);

    parallelLinearSearch<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_arr, size, d_matchedIndices, d_numMatches, target);

    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(numMatches, d_numMatches, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(matchedIndices, d_matchedIndices, *numMatches * 2 * sizeof(ll), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    // Open the file for writing
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "id,value,symbol\n");

    // Write the results to the file
    if (*numMatches == 0) {
        fprintf(file, "Element not found.\n");
    } else {
        for (int i = 0; i < *numMatches; i++) {
            fprintf(file, "%s", data[matchedIndices[i]]);
        }
    }

    // Close the file
    fclose(file);

    cudaFree(d_arr);
    cudaFree(d_matchedIndices);
    cudaFree(d_numMatches);
    cudaStreamDestroy(stream);
}

void mergeSortWrapper(ll arr[], ll size, char **data_tokens, ll columnIndex) {
    ll *d_arr, *d_sorted;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc((void **)&d_arr, size * sizeof(ll));
    cudaMalloc((void **)&d_sorted, size * sizeof(ll));
    cudaMemcpyAsync(d_arr, arr, size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    ll numberOfBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (ll i = 2; i < size * 2; i *= 2) {
        parallelMergeSort<<<numberOfBlocks, BLOCK_SIZE, 0, stream>>>(d_arr, d_sorted, size, i);
        cudaStreamSynchronize(stream);
        // Swap
        ll *temp = d_arr;
        d_arr = d_sorted;
        d_sorted = temp;
    }
    cudaMemcpyAsync(arr, d_arr, size * sizeof(ll), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Open the file for writing
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "id,value,symbol\n");

    // Write the sorted results to the file
    bool *visited = (bool*)malloc(size * sizeof(bool));
    memset(visited, 0, size * sizeof(bool));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            ll num_tokens;
            char *tokens = strdup(data_tokens[j]);
            char** db_info = split(tokens, ',', &num_tokens);
            if (atoi(db_info[columnIndex]) == arr[i] && !visited[j]) {
                visited[j] = true;
                fprintf(file, "%s", data_tokens[j]);
                break;
            }
            free(tokens);
            for (int k = 0; k < num_tokens; k++) {
                free(db_info[k]);
            }
            free(db_info);
        }
    }

    // Close the file
    fclose(file);

    cudaFree(d_arr);
    cudaFree(d_sorted);
    cudaStreamDestroy(stream);
    free(visited);
}

void innerJoinWrapper(ll arr1[], ll arr2[], char **data1, char **data2, ll size1, ll size2) {
    ll *d_arr1, *d_arr2;
    int *numMatches = (int*)malloc(sizeof(int));
    *numMatches = 0;
    ll *matchedIndices = (ll*)malloc(MAX_MATCHES * 2 * sizeof(ll));
    ll *d_matchedIndices;
    int *d_numMatches;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_arr1, size1 * sizeof(ll));
    cudaMalloc(&d_arr2, size2 * sizeof(ll));
    cudaMalloc(&d_matchedIndices, MAX_MATCHES * 2 * sizeof(ll));
    cudaMalloc(&d_numMatches, sizeof(int));

    cudaMemcpyAsync(d_arr1, arr1, size1 * sizeof(ll), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_arr2, arr2, size2 * sizeof(ll), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_numMatches, numMatches, sizeof(int), cudaMemcpyHostToDevice, stream);

    ll numberOfBlocks = (size1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    parallelInnerJoin<<<numberOfBlocks, BLOCK_SIZE, 0, stream>>>(d_arr1, d_arr2, d_matchedIndices, d_numMatches, size1, size2);

    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(numMatches, d_numMatches, sizeof(ll), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(matchedIndices, d_matchedIndices, *numMatches * 2 * sizeof(ll), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    // Open the file for writing
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "id,value,symbol\n");

    // Write the join results to the file
    for (int i = 0; i < *numMatches; i++) {
        int index1 = matchedIndices[i * 2];
        int index2 = matchedIndices[(i * 2) + 1];
        fprintf(file, "%s", data1[index1]);
        fprintf(file, "%s", data2[index2]);
    }

    // Close the file
    fclose(file);

    free(numMatches);
    free(matchedIndices);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_matchedIndices);
    cudaFree(d_numMatches);
    cudaStreamDestroy(stream);
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
                ll target;
                printf("EQUAL TO ");
                scanf("%lld", &target);
                ll data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                ll size = readCSV(table, data_tokens, data, column, &columnIndex);
                
                // Parallel Linear Search
                linearSearchWrapper(data, data_tokens, size, target);

                free(data_tokens);
                break;
            }
            case 2: {
                char column[50], table[50];
                printf("SELECT * FROM ");
                scanf("%s", table);
                printf("ORDER BY ");
                scanf("%s", column);
                ll data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                ll size = readCSV(table, data_tokens, data, column, &columnIndex);
                
                // Parallel Merge Sort
                mergeSortWrapper(data, size, data_tokens, columnIndex);

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
                ll data1[MAX_VALUES], data2[MAX_VALUES], columnIndex1, columnIndex2;
                char** data_tokens1 = (char**)malloc(MAX_VALUES * sizeof(char*));
                char** data_tokens2 = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table1, ".csv");
                ll size1 = readCSV(table1, data_tokens1, data1, column1, &columnIndex1);
                strcat(table2, ".csv");
                ll size2 = readCSV(table2, data_tokens2, data2, column2, &columnIndex2);

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