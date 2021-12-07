#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initial_data(float* data, const int size)
{
    for(int i = 0; i < size; ++i){
        data[i] = (float)(rand()&0xFF);
    }
}

bool checkResult(float* a, float* b, const int nx, const int ny){
    double epsilon = 1.0E-8;

    for(int i = 0; i < nx; ++i){
        for(int j = 0; j < ny; ++j){
            if(abs(a[j*nx+i] - b[i*ny + j]) > epsilon) return false;
        }
    }
    return true;
}

__global__ void transByRow(float* in, float* out, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1<<11;
    int ny = 1<<11;
    int nElem = nx * ny;
    int nBytes = nElem * sizeof(int);

    int blockx = 16,blocky = 16;

    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);

    float* h_mat = (float *)malloc(nBytes);
    float* h_matT = (float *)malloc(nBytes);

    initial_data(h_mat, nElem);
    float *d_mat, *d_matT;
    CHECK(cudaMalloc((int **)& d_mat, nBytes));
    CHECK(cudaMalloc((int **)& d_matT, nBytes));
    CHECK(cudaMemcpy(d_mat, h_mat, nBytes, cudaMemcpyHostToDevice));

    transByRow<<<grid, block>>>(d_mat, d_matT, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_matT, d_matT, nBytes, cudaMemcpyDeviceToHost));
    if(checkResult(h_mat, h_matT, nx, ny)){
        printf("The result is correct\n");
    }
    CHECK(cudaDeviceReset());

    return 0;
}