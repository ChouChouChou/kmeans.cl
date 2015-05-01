//
//  main.c
//  kmeans
//
//  Created by LisaChou on 2015/3/31.
//  Copyright (c) 2015年 LisaChou. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define DCNT     2000 //資料個數
#define DIM      100  // 資料維度
#define K        3   // 叢聚個數
#define LOW      20  // 資料下限
#define UP       300 // 資料上限
#define MAX_ITER 20  /* 最大迭代   */
#define MIN_PT   0   /* 最小變動點 */


void   get_data();          // 取得資料
int     data[DCNT][DIM];    // 原始資料
void   kmeans_init();       // 演算法初始化
int     cent[DCNT][DIM];    // 選出來的重心
double  update_table(int* ch_pt); // 更新table
double  cal_dis(int *x, int *y, int *out,int dim);
int     cent_c[K];          /* 該叢聚資料數*/
double  dis_k[K][DIM];   /* 叢聚距離   */
int    table[DCNT];        /* 紀錄每個資料所屬叢聚是哪個k*/
void   update_cent();            // 更新重心位置
void   print_cent();             // 顯示重心位置


const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";



const char *KernelSource1 = "\n" \
"__kernel void cal_dis(                                                       \n" \
"   __global int *cl_x,                                              \n" \
"   __global  int *cl_y,                                             \n" \
"   __global  int *cl_out,                                             \n" \
"   const unsigned int dim)                                           \n" \
" {                                                 \n" \
"    int dimm = get_global_id(0);                   \n" \
"      //int cout[]={0,0,0};                \n" \
"      cl_out[dimm]=0;               \n" \
"    int t[dim];                                             \n" \
"        //printf(\"--cl_x[%d]= %d \\n\",dimm,cl_x[dimm] ) ;                               \n"\
"        //printf(\"==cl_y[%d] =%d \\n\", dimm,cl_y[dimm] ) ;                               \n"\
"    if(dimm < dim){                                    \n" \
"        t[dimm]=cl_x[dimm]-cl_y[dimm];                             \n" \
"        cl_out[dimm]+=t[dimm]*t[dimm];                                      \n" \
"        //printf(\"cl_out[%d] =%d \\n\", dimm, cl_out[dimm] ) ;                               \n"\
"                                             \n" \
"    }                                                  \n" \
"}                                                                      \n" \
"\n";

//err = clEnqueueReadBuffer(commands, cl_out, CL_TRUE, 0, sizeof(float) * dim, &min_dis, 0, NULL, NULL );




int main(int argc, const char * argv[])
{
    
    
    
    
    
    
    
    
    
    
    //my code   ///////////////////////  my code
    
    int    ch_pt;          /* 紀錄變動之點 */
    double sse2;           /* 此次迭代之sse */  //用來判斷收斂的
    double sse1;           /* 上一迭代之sse */
    int     iter=0;        /* 迭代計數器   */
    
    srand((unsigned)time(NULL));
    get_data();                      // step 0 - 取得資料
    //printf("測試資料：%d \n",data[0][1]);
    kmeans_init();                   /* step 1 - 初始化,隨機取得重心 */
    
    sse2 = update_table(&ch_pt);     /* step 2 - 更新一次對應表      */
    //sse2是當次累積的總重心距離
    //printf("隨機重心的總重心距離   = %.2lf\n", sse2);
    
    do{
        sse1 = sse2;
        ++iter;
        
        update_cent();             /* step 3 - 更新算出新的重心位置 （和第一次的隨機方法不同）            */
        
        sse2=update_table(&ch_pt); /* step 4 - 更新對應表          */
        //sse2是再算一次新的總重心距離   要跟剛存的sse1=sse2的sse1來比較用的
        printf("第 %d 次的總重心距離   = %.2lf",iter, sse2);
        
    }while(iter<MAX_ITER && sse1!=sse2 && ch_pt>MIN_PT); // 收斂條件
    
    //printf("印出最後重心位置：\n");
    //print_cent(); // 顯示最後重心位置
    
    printf("s   = %.2lf\n", sse2);
    printf("ch_pt = %d\n", ch_pt);
    printf("iter = %d\n", iter);
    
    //clWaitForEvents(1 , &event);
    
    
    return 0;
    
    
}//end  main()


////////////////////////////////////////////////////////////////////////////////////////////////////
// 取得資料，此處以隨機給
void get_data()
{
    int i, j;
    
    for(i=0; i<DCNT; ++i)
        for(j=0; j<DIM; ++j){
            data[i][j] = \
            LOW + (double)rand()*(UP-LOW) / RAND_MAX;
        }
    
    /*測試把所有data [i][j] 印出來
     for (i=0; i< DCNT;++i) {
     for(j=0; j<DIM; ++j){
     printf("%d ,",data[i][j]);}
     putchar('\n');}*/
    
}

// 演算化初始化  隨機取得重心
void   kmeans_init()
{
    int i, j, k, rnd;
    int pick[K];
    
    // 隨機找K 個不同資料點
    for(k=0; k<K; ++k){
        rnd = rand() % DCNT; // 隨機取一筆
        for(i=0; i<k && pick[i]!=rnd; ++i);
        if(i==k)
            pick[k]=rnd; // 沒重覆
        else
            --k; // 有重覆, 再找一次
    }
    
    
    
    // 將K 個資料點內容複制到重心cent
    //printf("第一次重心資料：\n");
    for(k=0; k<K; ++k){
        for(j=0; j<DIM; ++j){
            cent[k][j] = data[pick[k]][j];
            
            //printf("%6d ,",cent[k][j]);
        }
        putchar('\n');
    }
}

// 更新table, 傳回sse, 存入點之變動數
double update_table(int* ch_pt)
{
    int i, j, k, min_k;
    double t_sse=0.0 ;
    //double dis, min_dis ;
    //double  dis[DCNT];
    //double  min_dis[DCNT];
    int min_dis=0;
    int min_diss[DIM];
    int disss1[DIM];
    int dis1=0;
    int disss2[DIM];
    int dis2=0;
    int min=0;
    
    int err;
    cl_platform_id platform_id;
    cl_uint num_id;
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    //cl_mem cl_x;
    //cl_mem cl_y;
    //cl_mem cl_out;
    
    unsigned int dim = DIM ;
    
    //unsigned int dim = DIM ;
    cl_event event;
    
    *ch_pt=0;                          // 變動點數設0
    memset(cent_c, 0, sizeof(cent_c)); // 各叢聚資料數清0  cent_c[K]; /* 該叢聚資料數*/   memset 將cent_c中所有的值設定為0
    memset(dis_k, 0, sizeof(dis_k));   // 各叢聚距離和清0    dis_k[K][DIM]; /* 叢聚距離   */
    /////////
    
    err = clGetPlatformIDs(1, &platform_id, &num_id);
    if(err != CL_SUCCESS)
    {
        printf("Failed to get the ID of the platform (%i)\n", num_id);
        return EXIT_FAILURE;
    }
    //printf("Id of the platform: %i\n",num_id);
    
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        if(err == CL_INVALID_PLATFORM)
            printf("CL_INVALID_PLATFORM\n");
        if(err == CL_INVALID_DEVICE_TYPE)
            printf("CL_INVALID_DEVICE_TYPE\n");
        if(err == CL_INVALID_VALUE)
            printf("CL_INVALID_VALUE\n");
        if(err == CL_DEVICE_NOT_FOUND)
            printf("CL_DEVICE_NOT_FOUND\n");
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    // printf("\nOpenCL demo application started!\n");
    
    
    
    
    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command queue
    //cl_context context,cl_device_id device,cl_command_queue_properties properties,cl_int *errcode_ret)
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    //clFinish(commands);
    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource1, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernel in the program we wish to run
    //cl_kernel clCreateKernel (cl_program program,  const char *kernel_name, cl_int *errcode_ret)
    kernel = clCreateKernel(program, "cal_dis", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    
    // Create the input and output arrays in device memory for our calculation
    //
    cl_mem cl_xx  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * dim , NULL, NULL);
    cl_mem cl_yy = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * dim , NULL, NULL);
    
    cl_mem cl_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * dim , NULL, NULL);
    if (!cl_xx || !cl_yy)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    //////////
    //cl_int clEnqueueWriteBuffer( cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t cb, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
    // Write our data set into the input array in device memory
    // copy the input buffer to device
    
    // //min_dis[i] = cal_dis(data[i], cent[0], DIM);
    
    
    
    for(i=0; i<DCNT; ++i){
        //printf("####### %p",cl_xx);
        
        
        
        
        //printf("IIIIII=%d\n",i);
        err = clEnqueueWriteBuffer(commands, cl_xx, CL_TRUE, 0, sizeof(int) * dim, &data[i], 0, NULL, NULL);
        err = clEnqueueWriteBuffer(commands, cl_yy, CL_TRUE, 0, sizeof(int) * dim, &cent[0] , 0, NULL, NULL);
        
        
        
        //err = clEnqueueWriteBuffer(commands, cl_out, CL_TRUE, 0, sizeof(float) * DIM, out , 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
        
        // Set the arguments to our compute kernel
        //
        err = 0;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xx);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yy);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_out);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &dim);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }
        
        
        //size_t work_size = DCNT;
        size_t local_size = DIM;
        err = clEnqueueNDRangeKernel(commands, kernel, 1 ,NULL, &local_size, NULL , 0, 0, &event);
        if (err)
        {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }
        
        
        
        
        err = clEnqueueReadBuffer(commands, cl_out, CL_TRUE, 0, sizeof(int) * dim, &min_diss, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        //clFinish(commands);
        clFlush(commands);
        
        
        for (int i = 0; i < dim; i++){
            //printf("%d : %d \n",i, min_diss[i]);
            min_dis+=min_diss[i];
        }
        //printf("min_dis =  %d \n",min_dis);
        
        
        
        //針對每一行去找他的k是誰跟最小距離是多少
        //for(i=0; i<DCNT; ++i){
        
        // 尋找所屬重心
        //先把每行跟第一個k值比的結果設為最小  之後再跟第二 第三k值比較
        
        
        
        //min_dis[i] = cal_dis(data[i], cent[0], DIM);
        
        min_k   = 0;
        for(k=1;k<K; ++k){
            if(k==1){
                
                err = clEnqueueWriteBuffer(commands, cl_xx, CL_TRUE, 0, sizeof(int) * dim, &data[i], 0, NULL, NULL);
                err = clEnqueueWriteBuffer(commands, cl_yy, CL_TRUE, 0, sizeof(int) * dim, &cent[k] , 0, NULL, NULL);
                
                
                
                //err = clEnqueueWriteBuffer(commands, cl_out, CL_TRUE, 0, sizeof(float) * DIM, out , 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to write to source array!\n");
                    exit(1);
                }
                
                // Set the arguments to our compute kernel
                //
                err = 0;
                err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xx);
                err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yy);
                err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_out);
                err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &dim);
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to set kernel arguments! %d\n", err);
                    exit(1);
                }
                
                
                size_t work_size = DIM;
                err = clEnqueueNDRangeKernel(commands, kernel, 1, 0, &work_size, 0, 0, 0, &event);
                if (err)
                {
                    printf("Error: Failed to execute kernel!\n");
                    return EXIT_FAILURE;
                }
                
                //clWaitForEvents(1 , &event);
                
                
                
                err = clEnqueueReadBuffer( commands, cl_out, CL_TRUE, 0, sizeof(float) * DIM, &disss1, 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to read output array! %d\n", err);
                    exit(1);
                }
                //clFinish(commands);
                clFlush(commands);
                //printf("dis = %f",dis);
                
                for (int i = 0; i < dim; i++){
                    //printf("%d : %d \n",i, disss1[i]);
                    dis1+=disss1[i];
                }
                //printf("disssss1 =  %d \n",dis1);
                
            }//end k==1
            k++;
            if(k==2){
                err = clEnqueueWriteBuffer(commands, cl_xx, CL_TRUE, 0, sizeof(int) * dim, &data[i], 0, NULL, NULL);
                err = clEnqueueWriteBuffer(commands, cl_yy, CL_TRUE, 0, sizeof(int) * dim, &cent[k] , 0, NULL, NULL);
                
                
                
                //err = clEnqueueWriteBuffer(commands, cl_out, CL_TRUE, 0, sizeof(float) * DIM, out , 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to write to source array!\n");
                    exit(1);
                }
                
                // Set the arguments to our compute kernel
                //
                err = 0;
                err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xx);
                err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yy);
                err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_out);
                err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &dim);
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to set kernel arguments! %d\n", err);
                    exit(1);
                }
                
                
                size_t work_size = DIM;
                err = clEnqueueNDRangeKernel(commands, kernel, 1, 0, &work_size, 0, 0, 0, &event);
                if (err)
                {
                    printf("Error: Failed to execute kernel!\n");
                    return EXIT_FAILURE;
                }
                
                //clWaitForEvents(1 , &event);
                
                
                
                err = clEnqueueReadBuffer( commands, cl_out, CL_TRUE, 0, sizeof(float) * DIM, &disss2, 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    printf("Error: Failed to read output array! %d\n", err);
                    exit(1);
                }
                //clFinish(commands);
                clFlush(commands);
                //printf("dis = %f",dis);
                
                for (int i = 0; i < dim; i++){
                    //printf("%d : %d \n",i, disss2[i]);
                    dis2+=disss2[i];
                }
                //printf("disssss2 =  %d \n",dis2);
                
                
                
                
                
                
                
            }//end k==2
            
            
            //之後再跟第二 第三k值比較
            //min_k   = 0;
            //for(k=1;k<K; ++k){
            //dis[i] = cal_dis(data[i], cent[k], DIM);
            /*if(dis < min_dis){  //先把每行跟第一個k值比的結果設為最小  之後再跟第二 第三k值比較
             printf("-----------");
             min_dis=dis;
             min_k = k;
             }
             */
            
            if (min_dis<dis1&&min_dis<dis2) {
                min = min_dis;
                min_k = 0;
            }
            else if (dis1<min_dis&&dis1<dis2) {
                min = dis1;
                min_k = 1;
                
            }
            else if (dis2<min_dis&&dis2<dis1) {
                min = dis2;
                min_k = 2;
            }
            /*
             printf("min_dis ====== %d\n",min_dis);
             printf("min_dis ====== %d\n",dis1);
             printf("min_dis ====== %d\n",dis2);
             printf("min ====== %d\n",min);
             */
            
            
            
        }//end small for
        
        
        *ch_pt+=(table[i]!=min_k); // 更新變動點數   table[DCNT]; /* 紀錄每個資料所屬叢聚是哪個k*/
        table[i] = min_k;          // 更新每行資料所屬重心是哪一個k
        
        ++cent_c[min_k];           // 累計該重心資料數
        //printf("cent_c1:%d : %d :%d \n",cent_c[0],cent_c[1],cent_c[2]);
        
        t_sse += min;          // 累計總重心距離  >>t_sse用來判斷最後是否收斂的！！！！！！
        min_dis=0;
        dis1=0;
        dis2=0;
        
        for(j=0; j<DIM; ++j)       // 更新各叢聚總距離
            dis_k[min_k][j]+=data[i][j];    //dis_k[K][DIM];   /* 叢聚距離   */
        
        /*
         for (k=0; k<K; k++) {
         for (j=0; j<DIM; j++) {
         printf("dis_k = %f " , dis_k[k][j]);
         }
         putchar('\n');
         }*/
        
    }//end big for
    clFlush(commands);
    
    
    
    //execution kernel time
    //
    cl_ulong time_start, time_end;
    double total_time;
    //cl_event event;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    total_time = time_end - time_start;
    printf("\nExecution time in milliseconds = %0.3f s\n", (total_time / 1000000.0) );
    
    
    
    
    return t_sse;
}//end function




// 計算二點距離

double cal_dis(int *x, int *y, int *out ,int dim)  //cal_dis(data[i], cent[0], DIM)
{
    int i;
    int t;
    int sum =0.0;
    for(i=0; i<dim; ++i){
        t=x[i]-y[i];
        sum+=t*t;
    }
    return sum;
}

// 更新新的重心位置  不是用隨機
void update_cent()
{
    int k, j;
    for(k=0; k<K; ++k)
        for(j=0; j<DIM; ++j)
            cent[k][j]=dis_k[k][j]/cent_c[k];   //用真正的各叢集總距離  除以 該叢集重心的資料數量
}

// 顯示最後重心位置
void   print_cent()
{
    int j, k;
    for(k=0; k<K; ++k) {
        for(j=0; j<DIM; ++j)
            printf("%6d ", cent[k][j]);
        putchar('\n');
    }
}


////////////////////////////////////////////////////////////
