#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define MAX_VALUES 100

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

// Binary search on array
int binarySearch(int arr[], int left, int right, int target) {
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// Linear search on array
int linearSearch(int arr[], int result[], int size, int target) {
    int index = -1;
    for (int i = 0; i < size; ++i) {
        if (arr[i] == target) {
            result[++index] = i;
        }
    }
    return index;
}

// Merge two sorted sub-arrays
void merge(int arr[], int left, int mid, int right) {
    int temp[right - left + 1];
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (int idx = 0; idx < k; ++idx) {
        arr[left + idx] = temp[idx];
    }
}

// Merge sort algorithm
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Radix sort algorithm
void radixSort(int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }

    for (int exp = 1; max / exp > 0; exp *= 10) {
        int output[size];
        int count[10] = {0};

        for (int i = 0; i < size; ++i) {
            count[(arr[i] / exp) % 10]++;
        }

        for (int i = 1; i < 10; ++i) {
            count[i] += count[i - 1];
        }

        for (int i = size - 1; i >= 0; --i) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        for (int i = 0; i < size; ++i) {
            arr[i] = output[i];
        }
    }
}

// Sequential inner join
void innerJoin(int arr1[], char **data1 , int size1, int arr2[], char **data2, int size2) {
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (arr1[i] == arr2[j]) {
                printf("%s%s\n", data1[i], data2[j]);
            }
        }
    }
}

int readCSV(char *filename, char **data, int arr[], char *column, int *columnIndex) {
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
                char column[50], table[50];
                printf("SELECT * FROM ");
                scanf("%s", table);
                printf("WHERE ");
                scanf("%s", column);
                int target;
                printf("EQUAL TO ");
                scanf("%d", &target);
                int data[MAX_VALUES], result[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                int size = readCSV(table, data_tokens, data, column, &columnIndex);
                int count = linearSearch(data, result, size, target);
                printf("Result:\n");
                if (count != -1) {
                    for (int i = 0; i <= count; i++) {
                        printf("%s", data_tokens[result[i]]);
                    }
                } else {
                    printf("Element not found.\n");
                }
                free(data_tokens);
                break;
            }
            case 2:
                char column[50], table[50];
                printf("SELECT * FROM ");
                scanf("%s", table);
                printf("ORDER BY ");
                scanf("%s", column);
                int data[MAX_VALUES], columnIndex;
                char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
                strcat(table, ".csv");
                int size = readCSV(table, data_tokens, data, column, &columnIndex);
                mergeSort(data, 0, size - 1);
                printf("Result:\n");
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

                // Perform inner join operation
                printf("Result:\n");
                innerJoin(data1, data_tokens1, size1, data2, data_tokens2, size2);
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