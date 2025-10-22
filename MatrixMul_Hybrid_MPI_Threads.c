///////////////////////////////////////////////////////////////////////////////////////////
// MatrixMul_Hybrid_MPI_Threads.c
// --------------------------------------------------------------------------------------
// Hybrid MPI + POSIX Threads Matrix Multiplication with Tile-Based Partitioning
// 
// Design:
// 1. Root process reads matrices using parallel threads
// 2. Root distributes only required tiles to each MPI process
// 3. Each MPI process uses threads to compute its assigned tiles
// 4. Root gathers results and writes to output file
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <pthread.h>
#include <math.h>

// Structure for thread arguments during file reading
typedef struct {
    FILE *file;
    int *matrix;
    int start_row;
    int end_row;
    int cols;
    pthread_mutex_t *file_mutex;
} ReadThreadArgs;

// Structure for thread arguments during computation
typedef struct {
    int *matrixA;
    int *matrixB;
    unsigned long long *matrixC;
    int start_row;
    int end_row;
    int rowA;
    int colA;
    int colB;
} ComputeThreadArgs;

// Thread function for parallel file reading
void* read_matrix_thread(void* arg) {
    ReadThreadArgs *args = (ReadThreadArgs*)arg;
    
    for (int i = args->start_row; i < args->end_row; i++) {
        pthread_mutex_lock(args->file_mutex);
        // Seek to the correct position: skip header (2 ints) + previous rows
        fseek(args->file, 2 * sizeof(int) + i * args->cols * sizeof(int), SEEK_SET);
        fread(&args->matrix[i * args->cols], sizeof(int), args->cols, args->file);
        pthread_mutex_unlock(args->file_mutex);
    }
    
    return NULL;
}

// Thread function for parallel matrix multiplication
void* compute_matrix_thread(void* arg) {
    ComputeThreadArgs *args = (ComputeThreadArgs*)arg;
    
    for (int i = args->start_row; i < args->end_row; i++) {
        for (int j = 0; j < args->colB; j++) {
            unsigned long long sum = 0;
            for (int k = 0; k < args->colA; k++) {
                sum += (unsigned long long)args->matrixA[i * args->colA + k] * 
                       (unsigned long long)args->matrixB[k * args->colB + j];
            }
            args->matrixC[i * args->colB + j] = sum;
        }
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int provided;
    
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 5) {
        if (rank == 0) {
            printf("Usage: %s <MatrixA_File> <MatrixB_File> <MatrixC_Output_File> <Threads_Per_Process>\n", argv[0]);
            printf("Example: mpirun -np 2 %s MA_1000x1000.bin MB_1000x1000.bin MC_1000x1000.bin 2\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    // get the files and number of threads
    char *matrixA_Filename = argv[1];
    char *matrixB_Filename = argv[2];
    char *matrixC_Filename = argv[3];
    int num_threads = atoi(argv[4]);
    // do not the thread
    // then gracefully exit
    if (num_threads < 1) {
        if (rank == 0) {
            printf("Error: Number of threads must be at least 1\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    int rowA = 0, colA = 0, rowB = 0, colB = 0;
    int *pMatrixA = NULL;
    int *pMatrixB = NULL;
    unsigned long long *pMatrixC = NULL;
    // reserve more space for the result matrix
    
    struct timespec start, end;
    double time_taken;
    
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        printf("\n=== Hybrid MPI + POSIX Threads Matrix Multiplication ===\n");
        printf("MPI Processes: %d, Threads per process: %d\n\n", size, num_threads);
    }
    
    // ==================== ROOT PROCESS: READ MATRICES WITH THREADS ====================
    if (rank == 0) {
        printf("[Root] Reading Matrix A (%s) with %d threads - Start\n", matrixA_Filename, num_threads);
        
        FILE *pFileA = fopen(matrixA_Filename, "rb");
        if (pFileA == NULL) {
            printf("Error: File %s doesn't exist.\n", matrixA_Filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fread(&rowA, sizeof(int), 1, pFileA);
        fread(&colA, sizeof(int), 1, pFileA);
        
        pMatrixA = (int*)malloc(rowA * colA * sizeof(int));
        if (pMatrixA == NULL) {
            printf("Error: Memory allocation failed for Matrix A.\n");
            fclose(pFileA);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // use threads to parallelise the read on the matrix file
        // using mutex to lock the file
        pthread_t *read_threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        ReadThreadArgs *read_args = (ReadThreadArgs*)malloc(num_threads * sizeof(ReadThreadArgs));
        pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;
        
        // split loads on to the threads
        int rows_per_thread = rowA / num_threads;
        int remaining_rows = rowA % num_threads;
        
        for (int t = 0; t < num_threads; t++) {
            read_args[t].file = pFileA;
            read_args[t].matrix = pMatrixA;
            read_args[t].start_row = t * rows_per_thread + (t < remaining_rows ? t : remaining_rows);
            read_args[t].end_row = read_args[t].start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
            read_args[t].cols = colA;
            read_args[t].file_mutex = &file_mutex;
            
            pthread_create(&read_threads[t], NULL, read_matrix_thread, &read_args[t]);
        }
        
        for (int t = 0; t < num_threads; t++) {
            pthread_join(read_threads[t], NULL);
        }
        // free back the threads and mutex to memory
        free(read_threads);
        free(read_args);
        pthread_mutex_destroy(&file_mutex);
        fclose(pFileA);
        printf("[Root] Reading Matrix A - Done\n");
        
        // Read Matrix B with threads
        printf("[Root] Reading Matrix B (%s) with %d threads - Start\n", matrixB_Filename, num_threads);
        
        FILE *pFileB = fopen(matrixB_Filename, "rb");
        if (pFileB == NULL) {
            printf("Error: File %s doesn't exist.\n", matrixB_Filename);
            free(pMatrixA);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fread(&rowB, sizeof(int), 1, pFileB);
        fread(&colB, sizeof(int), 1, pFileB);
        
        if (colA != rowB) {
            printf("Error: Matrix dimensions incompatible. colA(%d) != rowB(%d)\n", colA, rowB);
            fclose(pFileB);
            free(pMatrixA);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        pMatrixB = (int*)malloc(rowB * colB * sizeof(int));
        if (pMatrixB == NULL) {
            printf("Error: Memory allocation failed for Matrix B.\n");
            fclose(pFileB);
            free(pMatrixA);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        read_threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        read_args = (ReadThreadArgs*)malloc(num_threads * sizeof(ReadThreadArgs));
        pthread_mutex_init(&file_mutex, NULL);
        
        rows_per_thread = rowB / num_threads;
        remaining_rows = rowB % num_threads;
        
        for (int t = 0; t < num_threads; t++) {
            read_args[t].file = pFileB;
            read_args[t].matrix = pMatrixB;
            read_args[t].start_row = t * rows_per_thread + (t < remaining_rows ? t : remaining_rows);
            read_args[t].end_row = read_args[t].start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
            read_args[t].cols = colB;
            read_args[t].file_mutex = &file_mutex;
            
            pthread_create(&read_threads[t], NULL, read_matrix_thread, &read_args[t]);
        }
        
        for (int t = 0; t < num_threads; t++) {
            pthread_join(read_threads[t], NULL);
        }
        
        free(read_threads);
        free(read_args);
        pthread_mutex_destroy(&file_mutex);
        fclose(pFileB);
        printf("[Root] Reading Matrix B - Done\n");
        
        // Allocate result matrix
        pMatrixC = (unsigned long long*)calloc(rowA * colB, sizeof(unsigned long long));
        if (pMatrixC == NULL) {
            printf("Error: Memory allocation failed for Matrix C.\n");
            free(pMatrixA);
            free(pMatrixB);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // ==================== BROADCAST MATRIX DIMENSIONS ====================
    MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ==================== TILE-BASED DISTRIBUTION ====================
    // Calculate rows per process
    int rows_per_process = rowA / size;
    int remaining_rows = rowA % size;
    int my_start_row = rank * rows_per_process + (rank < remaining_rows ? rank : remaining_rows);
    int my_num_rows = rows_per_process + (rank < remaining_rows ? 1 : 0);
    
    if (rank == 0) {
        printf("\n[Root] Distributing tiles to %d MPI processes - Start\n", size);
    }
    
    // Allocate local matrices for non-root processes
    int *local_A = NULL;
    int *local_B = NULL;
    unsigned long long *local_C = NULL;
    
    if (rank != 0) {
        local_A = (int*)malloc(my_num_rows * colA * sizeof(int));
        local_B = (int*)malloc(rowB * colB * sizeof(int));
        local_C = (unsigned long long*)calloc(my_num_rows * colB, sizeof(unsigned long long));
        
        if (local_A == NULL || local_B == NULL || local_C == NULL) {
            printf("[Process %d] Error: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        local_A = pMatrixA + my_start_row * colA;
        local_B = pMatrixB;
        local_C = pMatrixC + my_start_row * colB;
    }
    
    // Root sends tiles to other processes
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            int p_start_row = p * rows_per_process + (p < remaining_rows ? p : remaining_rows);
            int p_num_rows = rows_per_process + (p < remaining_rows ? 1 : 0);
            
            // Send tile of Matrix A
            MPI_Send(&pMatrixA[p_start_row * colA], p_num_rows * colA, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
        
        // Broadcast entire Matrix B (all processes need it)
        MPI_Bcast(pMatrixB, rowB * colB, MPI_INT, 0, MPI_COMM_WORLD);
        printf("[Root] Distribution complete\n");
    } else {
        // Receive tile of Matrix A
        MPI_Recv(local_A, my_num_rows * colA, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive Matrix B via broadcast
        MPI_Bcast(local_B, rowB * colB, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // ==================== PARALLEL COMPUTATION WITH THREADS ====================
    if (rank == 0) {
        printf("\n[All Processes] Matrix Multiplication - Start\n");
    }
    
    pthread_t *compute_threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ComputeThreadArgs *compute_args = (ComputeThreadArgs*)malloc(num_threads * sizeof(ComputeThreadArgs));
    
    int rows_per_thread = my_num_rows / num_threads;
    int remaining_thread_rows = my_num_rows % num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        compute_args[t].matrixA = local_A;
        compute_args[t].matrixB = (rank == 0) ? local_B : local_B;
        compute_args[t].matrixC = local_C;
        compute_args[t].start_row = t * rows_per_thread + (t < remaining_thread_rows ? t : remaining_thread_rows);
        compute_args[t].end_row = compute_args[t].start_row + rows_per_thread + (t < remaining_thread_rows ? 1 : 0);
        compute_args[t].rowA = my_num_rows;
        compute_args[t].colA = colA;
        compute_args[t].colB = colB;
        
        pthread_create(&compute_threads[t], NULL, compute_matrix_thread, &compute_args[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(compute_threads[t], NULL);
    }
    
    free(compute_threads);
    free(compute_args);
    
    if (rank == 0) {
        printf("[All Processes] Matrix Multiplication - Done\n");
    }
    
    // ==================== GATHER RESULTS AT ROOT ====================
    if (rank == 0) {
        printf("\n[Root] Gathering results from all processes - Start\n");
        
        for (int p = 1; p < size; p++) {
            int p_start_row = p * rows_per_process + (p < remaining_rows ? p : remaining_rows);
            int p_num_rows = rows_per_process + (p < remaining_rows ? 1 : 0);
            
            MPI_Recv(&pMatrixC[p_start_row * colB], p_num_rows * colB, MPI_UNSIGNED_LONG_LONG, 
                     p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        printf("[Root] Gathering complete\n");
    } else {
        // Send local result to root
        MPI_Send(local_C, my_num_rows * colB, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }
    
    // ==================== ROOT WRITES RESULT ====================
    if (rank == 0) {
        printf("\n[Root] Writing result to %s - Start\n", matrixC_Filename);
        
        FILE *pFileC = fopen(matrixC_Filename, "wb");
        if (pFileC == NULL) {
            printf("Error: Unable to create output file %s.\n", matrixC_Filename);
            free(pMatrixA);
            free(pMatrixB);
            free(pMatrixC);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        int rowC = rowA, colC = colB;
        fwrite(&rowC, sizeof(int), 1, pFileC);
        fwrite(&colC, sizeof(int), 1, pFileC);
        
        for (int i = 0; i < rowC; i++) {
            fwrite(&pMatrixC[i * colC], sizeof(unsigned long long), colC, pFileC);
        }
        
        fclose(pFileC);
        printf("[Root] Writing complete\n");
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_taken = (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
        
        printf("\n=== Results ===\n");
        printf("Matrix A: %d x %d\n", rowA, colA);
        printf("Matrix B: %d x %d\n", rowB, colB);
        printf("Matrix C: %d x %d\n", rowC, colC);
        printf("Overall time (s): %.6lf\n", time_taken);
        printf("\nHybrid MPI + POSIX Threads Matrix Multiplication - Done\n\n");
    }
    
    // ==================== CLEANUP ====================
    if (rank == 0) {
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixC);
    } else {
        free(local_A);
        free(local_B);
        free(local_C);
    }
    
    MPI_Finalize();
    return 0;
}
