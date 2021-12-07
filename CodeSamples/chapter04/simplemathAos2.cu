#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define LEN 1<<2

struct innerstruct
{
    float x;
    float y;
};

void initstruct(innerstruct* is, int size){
    for(int i = 0; i < size; ++i){
        is[i].x = (float)(rand()&0xFF);
        is[i].y = (float)(rand()&0xFF);
    }
}

__global__ void testinnerStruct(innerstruct *A, innerstruct *C, const int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        C[i].x = A[i].x + 2.0f;
        C[i].y = A[i].y + 3.0f;
    }
    return;
}

int main(){
    int nElem = LEN;
    int nBytes = nElem * sizeof(innerstruct);
    innerstruct *h_A = (innerstruct *)malloc(nBytes);
    initstruct(h_A, nElem);
    innerstruct *d_A, *d_C;
    CHECK(cudaMalloc((innerstruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerstruct**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    int blocksize = 128;
    dim3 block(blocksize,1);
    dim3 grid((nElem + block.x - 1)/ block.x, 1);

    testinnerStruct<<<grid, block>>>(d_A, d_C, nElem);

    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_A));
    free(h_A);
    return 0;
    // initstruct *d_A;
    
}