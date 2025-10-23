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
    ReadThreadArgs *arguments = (ReadThreadArgs*)arg; // cast generic pointer to ReadThreadArgs, so it can access its parameters. 

    // for loop 
    // loops over subset of rows this thread is responsible for reading from disk 

    for (int index = arguments->start_row; index < arguments->end_row; index++) {

        // check lock state 
        // if unlocked - locks it immediately 
        // if locked, calling thread doesn't proceed. 

        pthread_mutex_lock(arguments->file_mutex);

        // Seek to the correct position: skip header (2 ints) + previous rows
        // fseek - move the file's read/write cursor to start of desired row. 
        fseek(arguments->file, 2 * sizeof(int) + index * arguments->cols * sizeof(int), SEEK_SET);

        // reads one entire row of integers from file into memory 
        fread(&arguments->matrix[index * arguments->cols], sizeof(int), arguments->cols, arguments->file);

        // releases the mutext that the current thread previously locked. 
        // marks the mutex as avaliable 
        pthread_mutex_unlock(arguments->file_mutex);
    }
    
    return NULL;
}

// Thread function for parallel matrix multiplication
void* compute_matrix_thread(void* arg) {
    ComputeThreadArgs *arguments = (ComputeThreadArgs*)arg;
    
    // Compute A x B 
    // outer loop goes through rows of A and C which this thread is responsible for 
    for (int index = arguments->start_row; index < arguments->end_row; index++) {

        // middle loop goes through each column of B 
        for (int j = 0; j < arguments->colB; j++) {

            unsigned long long sum = 0;

            // innermost loop goes through all the elements in row index of A and column j of B 
            // this is where dot product takes place 
            for (int k = 0; k < arguments->colA; k++) {

                //dot product 
                sum += (unsigned long long)arguments->matrixA[index * arguments->colA + k] * 
                       (unsigned long long)arguments->matrixB[k * arguments->colB + j];
            }
            // store the computed value into correct cell in result matrix C 
            arguments->matrixC[index * arguments->colB + j] = sum;
        }
    }
    
    //standard pthread convention 
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

        //read the row and column
        fread(&rowB, sizeof(int), 1, pFileB);
        fread(&colB, sizeof(int), 1, pFileB);

        //error catching
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

        //parallel file reading
        // allocate memory
        read_threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        read_args = (ReadThreadArgs*)malloc(num_threads * sizeof(ReadThreadArgs));
        pthread_mutex_init(&file_mutex, NULL);

        //distribute workload
        rows_per_thread = rowB / num_threads;
        remaining_rows = rowB % num_threads;

        //thread creation
        // each thread gets
        // File pointer and destination matrix
        // Range of rows to read
        // Mutex for synchronisation
        for (int t = 0; t < num_threads; t++) {
            read_args[t].file = pFileB;
            read_args[t].matrix = pMatrixB;
            read_args[t].start_row = t * rows_per_thread + (t < remaining_rows ? t : remaining_rows);
            read_args[t].end_row = read_args[t].start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
            read_args[t].cols = colB;
            read_args[t].file_mutex = &file_mutex;
            
            pthread_create(&read_threads[t], NULL, read_matrix_thread, &read_args[t]);
        }

        //synchronisation 
        for (int t = 0; t < num_threads; t++) {
            pthread_join(read_threads[t], NULL);
        }

        // free threads and memory mutexes
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
    // Calculate 2D process grid (as square as possible)
    int tile_rows = (int)sqrt(size);
    while (size % tile_rows != 0) tile_rows--;
    int tile_cols = size / tile_rows;
    
    // This process's position in 2D grid
    int tile_row_pos = rank / tile_cols;
    int tile_col_pos = rank % tile_cols;
    
    // Calculate tile dimensions for rows (Matrix A)
    int rows_per_process = rowA / tile_rows;
    int remaining_rows_A = rowA % tile_rows;
    int my_start_row = tile_row_pos * rows_per_process + (tile_row_pos < remaining_rows_A ? tile_row_pos : remaining_rows_A);
    int my_num_rows = rows_per_process + (tile_row_pos < remaining_rows_A ? 1 : 0);
    
    // Calculate tile dimensions for columns (Matrix B)
    int cols_per_process = colB / tile_cols;
    int remaining_cols_B = colB % tile_cols;
    int my_start_col = tile_col_pos * cols_per_process + (tile_col_pos < remaining_cols_B ? tile_col_pos : remaining_cols_B);
    int my_num_cols = cols_per_process + (tile_col_pos < remaining_cols_B ? 1 : 0);
    
    if (rank == 0) {
        printf("\n[Root] 2D Process Grid: %d x %d (Total: %d processes)\n", tile_rows, tile_cols, size);
    }
    
    // Allocate local matrices for ALL processes (including root)
    int *local_A = (int*)malloc(my_num_rows * colA * sizeof(int));
    int *local_B = (int*)malloc(rowB * my_num_cols * sizeof(int));
    unsigned long long *local_C = (unsigned long long*)calloc(my_num_rows * my_num_cols, sizeof(unsigned long long));
    
    if (local_A == NULL || local_B == NULL || local_C == NULL) { 
        printf("[Process %d] Error: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Root distributes tiles to ALL processes 
    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            int p_row = p / tile_cols;
            int p_col = p % tile_cols;
            
            // Calculate this process's tile boundaries
            int p_start_row = p_row * rows_per_process + (p_row < remaining_rows_A ? p_row : remaining_rows_A);
            int p_num_rows = rows_per_process + (p_row < remaining_rows_A ? 1 : 0);
            
            int p_start_col = p_col * cols_per_process + (p_col < remaining_cols_B ? p_col : remaining_cols_B);
            int p_num_cols = cols_per_process + (p_col < remaining_cols_B ? 1 : 0);
            
            printf("[Root] Process %d (grid[%d,%d]): rows[%d:%d], cols[%d:%d]\n", 
                   p, p_row, p_col, p_start_row, p_start_row + p_num_rows - 1,
                   p_start_col, p_start_col + p_num_cols - 1);
            
            // Extract tile of Matrix A (specific rows, all columns)
            int *tile_A = (int*)malloc(p_num_rows * colA * sizeof(int));
            for (int i = 0; i < p_num_rows; i++) {
                memcpy(&tile_A[i * colA], 
                       &pMatrixA[(p_start_row + i) * colA], 
                       colA * sizeof(int));
            }
            
            // Extract tile of Matrix B (all rows, specific columns)
            int *tile_B = (int*)malloc(rowB * p_num_cols * sizeof(int));
            for (int i = 0; i < rowB; i++) {
                memcpy(&tile_B[i * p_num_cols], 
                       &pMatrixB[i * colB + p_start_col], 
                       p_num_cols * sizeof(int));
            }
            
            if (p == 0) { 
                // Root copies to its own local memory
                memcpy(local_A, tile_A, p_num_rows * colA * sizeof(int));
                memcpy(local_B, tile_B, rowB * p_num_cols * sizeof(int));
            } else {
                // Send tiles to other processes)
                MPI_Send(tile_A, p_num_rows * colA, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(tile_B, rowB * p_num_cols, MPI_INT, p, 1, MPI_COMM_WORLD);
            }
            
            free(tile_A);
            free(tile_B);
        }
        printf("[Root] Distribution complete\n");
    } else {
        // Non-root processes receive their tiles
        MPI_Recv(local_A, my_num_rows * colA, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, rowB * my_num_cols, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        compute_args[t].matrixB = local_B;
        compute_args[t].matrixC = local_C;
        compute_args[t].start_row = t * rows_per_thread + (t < remaining_thread_rows ? t : remaining_thread_rows);
        compute_args[t].end_row = compute_args[t].start_row + rows_per_thread + (t < remaining_thread_rows ? 1 : 0);
        compute_args[t].rowA = my_num_rows;
        compute_args[t].colA = colA;
        compute_args[t].colB = my_num_cols; 
        
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
        
        for (int p = 0; p < size; p++) {
            int p_row = p / tile_cols;
            int p_col = p % tile_cols;
            
            int p_start_row = p_row * rows_per_process + (p_row < remaining_rows_A ? p_row : remaining_rows_A);
            int p_num_rows = rows_per_process + (p_row < remaining_rows_A ? 1 : 0);
            
            int p_start_col = p_col * cols_per_process + (p_col < remaining_cols_B ? p_col : remaining_cols_B);
            int p_num_cols = cols_per_process + (p_col < remaining_cols_B ? 1 : 0);
            
            unsigned long long *tile_C;
            
            if (p == 0) {
                // Root uses its own local_C
                tile_C = local_C;
            } else {
                // Receive from other processes
                tile_C = (unsigned long long*)malloc(p_num_rows * p_num_cols * sizeof(unsigned long long));
                MPI_Recv(tile_C, p_num_rows * p_num_cols, MPI_UNSIGNED_LONG_LONG, 
                         p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Place tile into correct position in final result matrix
            for (int i = 0; i < p_num_rows; i++) {
                memcpy(&pMatrixC[(p_start_row + i) * colB + p_start_col], 
                       &tile_C[i * p_num_cols], 
                       p_num_cols * sizeof(unsigned long long));
            }
            
            if (p != 0) {
                free(tile_C);
            }
        }
        
        printf("[Root] Gathering complete\n");
    } else {
        // Send local result to root
        MPI_Send(local_C, my_num_rows * my_num_cols, MPI_UNSIGNED_LONG_LONG, 0, 2, MPI_COMM_WORLD);
    }
    
    // ==================== ROOT WRITES RESULT ====================
    if (rank == 0) {
        // Only the root MPI process performs I/O
        printf("\nRoot: begin writing result to %s\n", matrixC_Filename);
        
        // Try to open the output file: writing in binary mode
        FILE *pFileC = fopen(matrixC_Filename, "wb");

        // If the file cannot be created/opened
        if (pFileC == NULL) {

            printf("Error: unable to create output file %s!!\n", matrixC_Filename);

            // Free allocated memory and then abort failed task
            free(pMatrixA);
            free(pMatrixB);
            free(pMatrixC);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Otherwise, get no. rows and columns of result matrix C
        int rowC = rowA, colC = colB;
        // Write dimensions for matrix C
        fwrite(&rowC, sizeof(int), 1, pFileC);
        fwrite(&colC, sizeof(int), 1, pFileC);
        
        // Write the actual data of matrix C
        for (int i = 0; i < rowC; i++) {
            // Write one row at a time
            fwrite(&pMatrixC[i * colC], sizeof(unsigned long long), colC, pFileC);
        }
        
        fclose(pFileC);
        printf("Root: writing is complete:D\n");
        
        // Calculate time taken
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_taken = (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
        
        // Print summary of results (and overall time taken)
        printf("\n=== Results ===\n");
        printf("Matrix A: %d x %d\n", rowA, colA);
        printf("Matrix B: %d x %d\n", rowB, colB);
        printf("Matrix C: %d x %d\n", rowC, colC);
        printf("2D Process Grid: %d x %d\n", tile_rows, tile_cols);
        printf("Overall time (s): %.6lf\n", time_taken);
        printf("\nHybrid MPI + POSIX Threads Matrix Multiplication - Done\n\n");
    }
    
    // ==================== CLEANUP ====================
    if (rank == 0) {
        // Root will free all three matrices
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixC);
    } else {
        // Non-root processes free only their local tile matrices
        free(local_A);
        free(local_B);
        free(local_C);
    }
    
    // Ends the MPI parallel environment
    MPI_Finalize();
    return 0;
}
