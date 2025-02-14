#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Structure to represent a matrix
typedef struct {
    int rows;
    int cols;
    int **data;
} Matrix;

// Function to allocate memory for a matrix and initialize it to zero
Matrix createMatrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;

    m.data = (int **)malloc(rows * sizeof(int *));
    if (m.data == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        m.data[i] = (int *)calloc(cols, sizeof(int));  // Initialize to zero
        if (m.data[i] == NULL) {
            perror("Memory allocation failed");
            exit(1);
        }
    }

    return m;
}

// Function to free the memory allocated for a matrix
void freeMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        free(m.data[i]);
    }
    free(m.data);
}

// Function to add two matrices
Matrix addMatrix(Matrix a, Matrix b) {
    Matrix result = createMatrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
}

// Function to subtract two matrices
Matrix subtractMatrix(Matrix a, Matrix b) {
    Matrix result = createMatrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

// Standard matrix multiplication for base cases (n ≤ 2)
Matrix multiplyStandard(Matrix a, Matrix b) {
    Matrix result = createMatrix(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            for (int k = 0; k < a.cols; k++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

// Function to pad a matrix to the next power of 2
int nextPowerOf2(int n) {
    return pow(2, ceil(log2(n)));
}

// Strassen's Matrix Multiplication
Matrix strassenMultiply(Matrix a, Matrix b) {
    int n = a.rows;
    
    // Base case: Use standard multiplication for small matrices
    if (n <= 2) {
        return multiplyStandard(a, b);
    }

    int newSize = n / 2;

    // Creating submatrices
    Matrix a11 = createMatrix(newSize, newSize);
    Matrix a12 = createMatrix(newSize, newSize);
    Matrix a21 = createMatrix(newSize, newSize);
    Matrix a22 = createMatrix(newSize, newSize);
    Matrix b11 = createMatrix(newSize, newSize);
    Matrix b12 = createMatrix(newSize, newSize);
    Matrix b21 = createMatrix(newSize, newSize);
    Matrix b22 = createMatrix(newSize, newSize);

    // Splitting the matrices into 4 submatrices
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            a11.data[i][j] = a.data[i][j];
            a12.data[i][j] = a.data[i][j + newSize];
            a21.data[i][j] = a.data[i + newSize][j];
            a22.data[i][j] = a.data[i + newSize][j + newSize];

            b11.data[i][j] = b.data[i][j];
            b12.data[i][j] = b.data[i][j + newSize];
            b21.data[i][j] = b.data[i + newSize][j];
            b22.data[i][j] = b.data[i + newSize][j + newSize];
        }
    }

    // Computing 7 products
    Matrix p1 = strassenMultiply(addMatrix(a11, a22), addMatrix(b11, b22));
    Matrix p2 = strassenMultiply(addMatrix(a21, a22), b11);
    Matrix p3 = strassenMultiply(a11, subtractMatrix(b12, b22));
    Matrix p4 = strassenMultiply(a22, subtractMatrix(b21, b11));
    Matrix p5 = strassenMultiply(addMatrix(a11, a12), b22);
    Matrix p6 = strassenMultiply(subtractMatrix(a21, a11), addMatrix(b11, b12));
    Matrix p7 = strassenMultiply(subtractMatrix(a12, a22), addMatrix(b21, b22));

    // Computing final submatrices
    Matrix c11 = addMatrix(subtractMatrix(addMatrix(p1, p4), p5), p7);
    Matrix c12 = addMatrix(p3, p5);
    Matrix c21 = addMatrix(p2, p4);
    Matrix c22 = addMatrix(subtractMatrix(addMatrix(p1, p3), p2), p6);

    // Combining results
    Matrix result = createMatrix(n, n);
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            result.data[i][j] = c11.data[i][j];
            result.data[i][j + newSize] = c12.data[i][j];
            result.data[i + newSize][j] = c21.data[i][j];
            result.data[i + newSize][j + newSize] = c22.data[i][j];
        }
    }

    // Free memory
    freeMatrix(a11); freeMatrix(a12); freeMatrix(a21); freeMatrix(a22);
    freeMatrix(b11); freeMatrix(b12); freeMatrix(b21); freeMatrix(b22);
    freeMatrix(p1); freeMatrix(p2); freeMatrix(p3); freeMatrix(p4);
    freeMatrix(p5); freeMatrix(p6); freeMatrix(p7);
    freeMatrix(c11); freeMatrix(c12); freeMatrix(c21); freeMatrix(c22);

    return result;
}

int main() {
    int n;
    printf("Enter the size of the square matrices: ");
    scanf("%d", &n);

    int paddedSize = nextPowerOf2(n); // Ensure the matrix size is a power of 2

    Matrix a = createMatrix(paddedSize, paddedSize);
    Matrix b = createMatrix(paddedSize, paddedSize);

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &a.data[i][j]);

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &b.data[i][j]);

    Matrix result = strassenMultiply(a, b);

    printf("Resultant matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%d ", result.data[i][j]);
        printf("\n");
    }

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(result);
    return 0;
}
