#include <stdio.h>

int main() {
    double sum = 0.0;
    for (int i = 0; i < 100000000; i++) {
        sum += i * 0.001;
    }
    printf("%f\n", sum);
    return 0;
}