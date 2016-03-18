#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<stdlib.h>


#define O_Tile_Width 3
#define Mask_width 3
#define width 5
#define Block_width (O_Tile_Width+(Mask_width-1))
#define Mask_radius (Mask_width/2)


__global__ void convolution_1D_tiled(float *N,float *M,float *P)
{
int index_out_x=blockIdx.x*O_Tile_Width+threadIdx.x;
int index_in_x=index_out_x-Mask_radius;
__shared__ float N_shared[Block_width];
float Pvalue=0.0;

//Load Data into shared Memory (into TILE)
if((index_in_x>=0)&&(index_in_x<width))
{
 N_shared[threadIdx.x]=N[index_in_x];
}
else
{
 N_shared[threadIdx.x]=0.0f;
}
__syncthreads();

//Calculate Convolution (Multiply TILE and Mask Arrays)
if(threadIdx.x<O_Tile_Width)
{
 //Pvalue=0.0f;
 for(int j=0;j<Mask_width;j++)
 {
  Pvalue+=M[j]*N_shared[j+threadIdx.x];
 }
 P[index_out_x]=Pvalue;
}


}

int main()
{
 float * input;
 float * Mask;
 float * output;

 float * device_input;
 float * device_Mask;
 float * device_output;

 input=(float *)malloc(sizeof(float)*width);
 Mask=(float *)malloc(sizeof(float)*Mask_width);
 output=(float *)malloc(sizeof(float)*width);

 for(int i=0;i<width;i++)
 {
  input[i]=1.0;
 }

 for(int i=0;i<Mask_width;i++)
 {
  Mask[i]=1.0;
 }
  printf("\nInput:\n");
  for(int i=0;i<width;i++)
  {
   printf(" %0.2f\t",*(input+i));
  }
  printf("\nMask:\n");
   for(int i=0;i<Mask_width;i++)
   {
    printf(" %0.2f\t",*(Mask+i));
   }

 cudaMalloc((void **)&device_input,sizeof(float)*width);
 cudaMalloc((void **)&device_Mask,sizeof(float)*Mask_width);
 cudaMalloc((void **)&device_output,sizeof(float)*width);

 cudaMemcpy(device_input,input,sizeof(float)*width,cudaMemcpyHostToDevice);
 cudaMemcpy(device_Mask,Mask,sizeof(float)*Mask_width,cudaMemcpyHostToDevice);

 dim3 dimBlock(Block_width,1,1);
 dim3 dimGrid((((width-1)/O_Tile_Width)+1),1,1);
 convolution_1D_tiled<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output);

 cudaMemcpy(output,device_output,sizeof(float)*width,cudaMemcpyDeviceToHost);

 printf("\nOutput:\n");
 for(int i=0;i<width;i++)
 {
  printf(" %0.2f\t",*(output+i));
 }

 cudaFree(device_input);
 cudaFree(device_Mask);
 cudaFree(device_output);
 free(input);
 free(Mask);
 free(output);

printf("\n\nNumber of Blocks: %d ",dimGrid.x);
printf("\n\nNumber of Threads Per Block: %d ",dimBlock.x);

return 0;
}
