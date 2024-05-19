#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>


int getNextValue(int x){
  int i =1;
  while(i<x){
    i*=2;
  }
  return i;
}

int ORIGINAL_NUMBER=100000;
int MAX_VALUES = getNextValue(ORIGINAL_NUMBER);
#define MAX_THREADS_PER_BLOCK 128;

typedef long long ll;

void bitonicSortCPU(int* arr, int* indices, int n) 
{
    for (int k = 2; k <= n; k *= 2) 
    {
        for (int j = k / 2; j > 0; j /= 2) 
        {
            for (int i = 0; i < n; i++) 
            {
                int ij = i ^ j;

                if ((i & k) == 0)
                {
                    if (arr[i] > arr[ij])
                    {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;

                        int tempIdx = indices[i];
                        indices[i] = indices[ij];
                        indices[ij] = tempIdx;
                    }
                }
                else
                {
                    if (arr[i] < arr[ij])
                    {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;

                        int tempIdx = indices[i];
                        indices[i] = indices[ij];
                        indices[ij] = tempIdx;
                    }
                }
            }
        }
    }
}

// Function to split a string by delimiter
char** split(const char* s, char delimiter, ll* num_tokens) {
    //we need to get the tokens of the string separated by a delimiter
    //then this will be the array of tokens that we have
    //then generate the tokens array
    char** tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
    if (tokens == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    //move forward after the , as this is the shape of our csv file
    char* token = strtok((char*)s, ",");
    //initialize number of tokens with 0 then increment it while we keep moving forward
    *num_tokens = 0;
    //Keep moving forward and extract each token as each one will be separated by a comma then we will need to apply strtok to move forward
    while (token != NULL) {
        //fill the current tokens array
         tokens[*num_tokens] = strdup(token);
        if (tokens[*num_tokens] == NULL) {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        //increment number of token as we use it to get the size of the lines and number of columns in another way
        (*num_tokens)++;
        //keep moving forward
        token = strtok(NULL, ",");
    }
    return tokens;
}

// Function to read a CSV file
ll readCSV(const char *filename, char **data, ll arr[], char *column, ll *columnIndex) {
    //Initilaize the file than contains our data    
    FILE* file = fopen(filename, "r");
    //Size of the file which will describe the number of lines we have
    int size = 0;
    //Data of a single line which will be maximum of 1024 char
    char line[1024];
    //boolean used to check if we are in the first line or not
    bool firstLine = true;
    ll num_tokens;
      //looping over the file , line by line and extract the data from it
    while (fgets(line, sizeof(line), file)) {
        //If we are in the first line then we will have a different behaviour
        if (firstLine) {
            //Extract all the token existed from that lines and store their count in num_tokens
            char** tokens = split(line, ',', &num_tokens);
            //Then Loop for Each token which will be in our example here 3 tokens
            //We Want to get the index of the column we are interested in
            //and then store it in columnIndex then now we know where we will look at
            for (int i = 0; i < num_tokens; i++) {
                if (strncmp(column, tokens[i], strlen(column)) == 0) {
                    *columnIndex = i;
                    break;
                }
            }
            //Change it to false to prevent us from getting to the point of choosing the index again
            //as a kind of optimization
            firstLine = false;
            continue;
        }
        //extract the current line and store it in the data array
        data[size] = strdup(line);
        //extract the tokens we want from that line
        char** tokens = split(line, ',', &num_tokens);
        //create the array of the values of column we are interested in
        arr[size++] = atoi(tokens[*columnIndex]);
        free(tokens);
    }
    int final_idx = getNextValue(size);
    while(size<final_idx){
      char output[300];
      sprintf(output,"%lld,%lld,%lld",LLONG_MIN,LLONG_MIN,LLONG_MIN );
      data[size] = strdup(output);//extract the tokens we want from that line
      char** tokens = split(output, ',', &num_tokens);
        //create the array of the values of column we are interested in
      arr[size++] = atoi(tokens[*columnIndex]);
      free(tokens);
    }
    fclose(file);
    return size;
}




__global__ void bitonicSortGPU(int* arr,int* indices, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;

                int tempIdx = indices[i];
                indices[i] = indices[ij];
                indices[ij] = tempIdx;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;

                int tempIdx = indices[i];
                indices[i] = indices[ij];
                indices[ij] = tempIdx;
            }
        }
    }
}

//Function to print array
void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

//Automated function to check if array is sorted
bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

//Function to check if given number is a power of 2
bool isPowerOfTwo(int num) 
{
    return num > 0 && (num & (num - 1)) == 0;
}

void printStringArray(char** array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%s\n", array[i]);
    }
    return;
}

void printLongLongArray(long long array[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%lld\n", array[i]);
    }
    return;
}

void extractColumnValues(char **stringArray, int numRows, int columnIndex, int *resultArray) {
    char *token;
    const char delimiter[2] = ",";
    
    for (int i = 0; i < numRows; i++) {
        // Make a copy of the current string because strtok modifies the string it processes
        char *stringCopy = strdup(stringArray[i]);
        int currentIndex = 0;
        
        // Use strtok to split the string by the delimiter
        token = strtok(stringCopy, delimiter);
        
        while (token != NULL) {
            if (currentIndex == columnIndex) {
                resultArray[i] = atoi(token);
                break;
            }
            token = strtok(NULL, delimiter);
            currentIndex++;
        }
        
        free(stringCopy); // Free the copy of the string
    }
}


//MAIN PROGRAM
int main()
{   
    char column[50], table[50];
    printf("SELECT * FROM ");
    scanf("%s", table);
    printf("ORDER BY ");
    scanf("%s", column);
    ll data[MAX_VALUES], columnIndex;
    char** data_tokens = (char**)malloc(MAX_VALUES * sizeof(char*));
    strcat(table, ".csv");
    ll size = readCSV(table, data_tokens, data, column, &columnIndex);
    //Al Array ely ana muhtm beha heya el data_tokens

    //Create CPU based Arrays
    int* arr = new int[size];
    extractColumnValues(data_tokens,size,columnIndex,arr);

    int* indices = new int[size];
    for ( int i =0;i<size;i++){
      indices[i]=i;
    }


    int* gpuArr;
    int* gpuIndices;

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMalloc((void**)&gpuArr, size * sizeof(int));
    cudaMalloc((void**)&gpuIndices, size * sizeof(int));

    cudaMemcpy(gpuArr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuIndices, indices, size * sizeof(int), cudaMemcpyHostToDevice);

    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);


    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;

    //Initialize CPU clock counters
    clock_t startCPU, endCPU;

    //Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int j, k;

    //Time the run and call GPU Bitonic Kernel
    cudaEventRecord(startGPU);
    for (k = 2; k <= size; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            bitonicSortGPU << <blocksPerGrid, threadsPerBlock >> > (gpuArr,gpuIndices, j, k);
        }
    }
    cudaEventRecord(stopGPU);

    //Transfer Sorted array back to CPU
    cudaMemcpy(indices, gpuIndices, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr, gpuArr, size * sizeof(int), cudaMemcpyDeviceToHost);
    FILE *file = fopen("GPU_sort_output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    cudaStreamSynchronize(stream2);
    for (int i = 0; i < size; i++) {
      if(indices[i]<ORIGINAL_NUMBER){
      fprintf(file, "%s", data_tokens[indices[i]]);
      }
    }
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

//------------------------------------------------------------------------------------------------------------------------------------------------------
    int* carr = new int[size];    
    extractColumnValues(data_tokens,size,columnIndex,carr);
    int* indices_CPU = new int[size];
    for ( int i =0;i<size;i++){
      indices_CPU[i]=i;
    }
    //Time the run and call CPU Bitonic Sort
    startCPU = clock();
    bitonicSortCPU(carr,indices_CPU, size);
    endCPU = clock();

    FILE *file2 = fopen("CPU_sort_output.txt", "w");
    if (file2 == NULL) {
        perror("Error opening file");
        return 1;
    }
    for (int i = 0; i < size; i++) {
      if(indices_CPU[i]<ORIGINAL_NUMBER){
      fprintf(file2, "%s", data_tokens[indices_CPU[i]]);
      }
    }
    //Calculate Elapsed CPU time
    double millisecondsCPU = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);

    //Print the time of the runs
    std::cout << "\n\nGPU Time: " << millisecondsGPU << " ms" << std::endl;
    std::cout << "CPU Time: " << millisecondsCPU << " ms" << std::endl;

    //Destroy all variables
    delete[] arr;
    delete[] carr;

    cudaFree(gpuArr);
    std::cout << "\n------------------------------------------------------------------------------------\n||||| END. YOU MAY RUN THIS AGAIN |||||\n------------------------------------------------------------------------------------";
    return 0;
}