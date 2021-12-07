#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define LEN 1<<22

struct innerstruct
{
    float x;
    float y;
};

struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initstruct(innerArray *is, int size)
{
    for(int i = 0; i < size; ++i){
        is->x[i] = (float)(rand()&0xFF);
        is->y[i] = (float)(rand()&0xFF);
    }
}

__global__ void testinnerArray(innerArray *A, innerArray *C, const int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        C->x[i] = A->x[i] + 2.0f;
        C->y[i] = A->y[i] + 3.0f;
    }
    return;
}


int main(){
    int nElem = LEN;
    int nBytes = sizeof(innerArray);
    innerArray *h_A = (innerArray *)malloc(nBytes);
    initstruct(h_A, nElem);
    innerArray *d_A, *d_C;
    CHECK(cudaMalloc((innerArray**)&d_A, nBytes));
    CHECK(cudaMalloc((innerArray**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    int blocksize = 128;
    dim3 block(blocksize,1);
    dim3 grid((nElem + block.x - 1)/ block.x, 1);

    testinnerArray<<<grid, block>>>(d_A, d_C, nElem);

    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_A));
    free(h_A);
    return 0;
    // initstruct *d_A;
    
}