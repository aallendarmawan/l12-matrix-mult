///////////////////////////////////////////////////////////////////////////////////////////
// MatrixMul_Fox_MPI_OMP.c
// --------------------------------------------------------------------------------------
// Fox's Algorithm for Matrix Multiplication using Hybrid MPI + OpenMP
// 
// Design:
// 1. Assumes perfect square matrices and q×q process grid (p = q²)
// 2. Root reads matrices and distributes tiles to all processes
// 3. Each process performs q iterations of Fox's algorithm:
//    - Broadcast A tiles along rows
//    - Multiply using OpenMP for intra-tile parallelization
//    - Shift B tiles upward in columns
// 4. Root gathers and writes final result
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

// Function to perform matrix multiplication of tiles using OpenMP
void multiply_tiles(int *A, int *B, unsigned long long *C, int tile_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < tile_size; i++) {
        for (int j = 0; j < tile_size; j++) {
            unsigned long long sum = 0;
            for (int k = 0; k < tile_size; k++) {
                sum += (unsigned long long)A[i * tile_size + k] * 
                       (unsigned long long)B[k * tile_size + j];
            }
            C[i * tile_size + j] += sum;
        }
    }
}

// Function to read matrix from binary file
int* read_matrix(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    fread(rows, sizeof(int), 1, file);
    fread(cols, sizeof(int), 1, file);
    
    int *matrix = (int*)malloc((*rows) * (*cols) * sizeof(int));
    if (matrix == NULL) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    for (int i = 0; i < *rows; i++) {
        fread(&matrix[i * (*cols)], sizeof(int), *cols, file);
    }
    
    fclose(file);
    return matrix;
}

// Function to write matrix to binary file
void write_matrix(const char *filename, unsigned long long *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }
    
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);
    
    for (int i = 0; i < rows; i++) {
        fwrite(&matrix[i * cols], sizeof(unsigned long long), cols, file);
    }
    
    fclose(file);
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
            printf("Usage: %s <MatrixA_File> <MatrixB_File> <MatrixC_Output_File> <OMP_Threads>\n", argv[0]);
            printf("Example: mpirun -np 4 %s MA_1000x1000.bin MB_1000x1000.bin MC_1000x1000.bin 2\n", argv[0]);
            printf("Note: Number of processes must be a perfect square (1, 4, 9, 16, ...)\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    char *matAName = argv[1];
    char *matBName = argv[2];
    char *matCName = argv[3];
    int num_threads = atoi(argv[4]);
    
    // Set OpenMP threads
    omp_set_num_threads(num_threads);
    
    // Check if size is a perfect square
    int q = (int)sqrt((double)size);
    if (q * q != size) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) must be a perfect square!\n", size);
            printf("Valid values: 1, 4, 9, 16, 25, 36, 49, 64, ...\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    int N = 0;  // Matrix dimension (N×N)
    int *pMatrixA = NULL;
    int *pMatrixB = NULL;
    unsigned long long *pMatrixC = NULL;
    
    struct timespec start, end;
    double time_taken;
    
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        printf("\n=== Fox's Algorithm: Hybrid MPI + OpenMP Matrix Multiplication ===\n");
        printf("MPI Processes: %d (%dx%d grid), OpenMP threads per process: %d\n\n", 
               size, q, q, num_threads);
    }
    
    // ==================== ROOT: READ MATRICES ====================
    if (rank == 0) {
        printf("[Root] Reading matrices - Start\n");
        
        int rowA, colA, rowB, colB;
        pMatrixA = read_matrix(matAName, &rowA, &colA);
        pMatrixB = read_matrix(matBName, &rowB, &colB);
        
        if (pMatrixA == NULL || pMatrixB == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Validate dimensions
        if (rowA != colA || rowB != colB || rowA != rowB) {
            printf("Error: Matrices must be square and same size for Fox's algorithm\n");
            printf("Matrix A: %dx%d, Matrix B: %dx%d\n", rowA, colA, rowB, colB);
            free(pMatrixA);
            free(pMatrixB);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (colA != rowB) {
            printf("Error: Matrix dimensions incompatible. colA(%d) != rowB(%d)\n", colA, rowB);
            free(pMatrixA);
            free(pMatrixB);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        N = rowA;
        
        if (N % q != 0) {
            printf("Error: Matrix size (%d) must be divisible by grid size (%d)\n", N, q);
            free(pMatrixA);
            free(pMatrixB);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        pMatrixC = (unsigned long long*)calloc(N * N, sizeof(unsigned long long));
        if (pMatrixC == NULL) {
            printf("Error: Memory allocation failed for Matrix C\n");
            free(pMatrixA);
            free(pMatrixB);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("[Root] Matrix size: %dx%d, Tile size: %dx%d\n", N, N, N/q, N/q);
        printf("[Root] Reading matrices - Done\n");
    }
    
    // ==================== BROADCAST MATRIX SIZE ====================
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int tile_size = N / q;
    
    // Calculate process position in grid
    int my_row = rank / q;
    int my_col = rank % q;
    
    // Allocate local tiles
    int *tile_A = (int*)malloc(tile_size * tile_size * sizeof(int));
    int *tile_B = (int*)malloc(tile_size * tile_size * sizeof(int));
    unsigned long long *tile_C = (unsigned long long*)calloc(tile_size * tile_size, 
                                                               sizeof(unsigned long long));
    int *temp_A = (int*)malloc(tile_size * tile_size * sizeof(int));
    
    if (tile_A == NULL || tile_B == NULL || tile_C == NULL || temp_A == NULL) {
        printf("[Process %d] Error: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // ==================== DISTRIBUTE INITIAL TILES ====================
    if (rank == 0) {
        printf("\n[Root] Distributing tiles to %dx%d process grid - Start\n", q, q);
        
        // Send tiles to all processes (including self)
        for (int proc = 0; proc < size; proc++) {
            int proc_row = proc / q;
            int proc_col = proc % q;
            
            // Extract tile from Matrix A
            int *temp_tile_A = (int*)malloc(tile_size * tile_size * sizeof(int));
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    int global_i = proc_row * tile_size + i;
                    int global_j = proc_col * tile_size + j;
                    temp_tile_A[i * tile_size + j] = pMatrixA[global_i * N + global_j];
                }
            }
            
            // Extract tile from Matrix B
            int *temp_tile_B = (int*)malloc(tile_size * tile_size * sizeof(int));
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    int global_i = proc_row * tile_size + i;
                    int global_j = proc_col * tile_size + j;
                    temp_tile_B[i * tile_size + j] = pMatrixB[global_i * N + global_j];
                }
            }
            
            if (proc == 0) {
                memcpy(tile_A, temp_tile_A, tile_size * tile_size * sizeof(int));
                memcpy(tile_B, temp_tile_B, tile_size * tile_size * sizeof(int));
            } else {
                MPI_Send(temp_tile_A, tile_size * tile_size, MPI_INT, proc, 0, MPI_COMM_WORLD);
                MPI_Send(temp_tile_B, tile_size * tile_size, MPI_INT, proc, 1, MPI_COMM_WORLD);
            }
            
            free(temp_tile_A);
            free(temp_tile_B);
        }
        
        printf("[Root] Distribution complete\n");
    } else {
        // Receive initial tiles
        MPI_Recv(tile_A, tile_size * tile_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(tile_B, tile_size * tile_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // ==================== FOX'S ALGORITHM ====================
    if (rank == 0) {
        printf("\n[All Processes] Fox's Algorithm - Start\n");
    }
    
    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);
    
    // Fox's algorithm: q iterations
    for (int stage = 0; stage < q; stage++) {
        // Step 1: Broadcast A tile along row
        int bcast_root = (my_row + stage) % q;
        
        if (my_col == bcast_root) {
            memcpy(temp_A, tile_A, tile_size * tile_size * sizeof(int));
        }
        
        MPI_Bcast(temp_A, tile_size * tile_size, MPI_INT, bcast_root, row_comm);
        
        // Step 2: Local multiplication using OpenMP
        multiply_tiles(temp_A, tile_B, tile_C, tile_size);
        
        // Step 3: Shift B tile upward in column (circular)
        int src = (my_row + 1) % q;
        int dest = (my_row - 1 + q) % q;
        
        int *temp_B = (int*)malloc(tile_size * tile_size * sizeof(int));
        
        MPI_Sendrecv(tile_B, tile_size * tile_size, MPI_INT, dest, 2,
                     temp_B, tile_size * tile_size, MPI_INT, src, 2,
                     col_comm, MPI_STATUS_IGNORE);
        
        memcpy(tile_B, temp_B, tile_size * tile_size * sizeof(int));
        free(temp_B);
    }
    
    if (rank == 0) {
        printf("[All Processes] Fox's Algorithm - Done\n");
    }
    
    // ==================== GATHER RESULTS ====================
    if (rank == 0) {
        printf("\n[Root] Gathering results - Start\n");
        
        // Collect tiles from all processes
        for (int proc = 0; proc < size; proc++) {
            int proc_row = proc / q;
            int proc_col = proc % q;
            
            unsigned long long *temp_tile_C = (unsigned long long*)malloc(tile_size * tile_size * 
                                                                          sizeof(unsigned long long));
            
            if (proc == 0) {
                memcpy(temp_tile_C, tile_C, tile_size * tile_size * sizeof(unsigned long long));
            } else {
                MPI_Recv(temp_tile_C, tile_size * tile_size, MPI_UNSIGNED_LONG_LONG, 
                        proc, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Place tile in result matrix
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    int global_i = proc_row * tile_size + i;
                    int global_j = proc_col * tile_size + j;
                    pMatrixC[global_i * N + global_j] = temp_tile_C[i * tile_size + j];
                }
            }
            
            free(temp_tile_C);
        }
        
        printf("[Root] Gathering complete\n");
    } else {
        // Send result tile to root
        MPI_Send(tile_C, tile_size * tile_size, MPI_UNSIGNED_LONG_LONG, 0, 3, MPI_COMM_WORLD);
    }
    
    // ==================== ROOT WRITES RESULT ====================
    if (rank == 0) {
        printf("\n[Root] Writing result to %s - Start\n", matCName);
        write_matrix(matCName, pMatrixC, N, N);
        printf("[Root] Writing complete\n");
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_taken = (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
        
        printf("\n=== Results ===\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Process grid: %d x %d\n", q, q);
        printf("Tile size: %d x %d\n", tile_size, tile_size);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Overall time (s): %.6lf\n", time_taken);
        printf("\nFox's Algorithm Matrix Multiplication - Done\n\n");
        
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixC);
    }
    
    // ==================== CLEANUP ====================
    free(tile_A);
    free(tile_B);
    free(tile_C);
    free(temp_A);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    
    MPI_Finalize();
    return 0;
}
