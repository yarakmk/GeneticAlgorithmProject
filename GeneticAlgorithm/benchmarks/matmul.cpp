#include <vector>
#include <iostream>

int main() {
    const int N = 200;

    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(N));
    std::vector<std::vector<double>> C(N, std::vector<double>(N));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = B[i][j] = i + j;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    std::cout << C[N-1][N-1] << "\n"; // prevent dead-code elimination
}
