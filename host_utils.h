#pragma once

#include "headers.h"


int getNegativeCounter(int* x, int *y, const int LEN) {

    int negative_counter = 0;
    for (int i = 0; i < LEN; i++) {
        if (x[i] < 0 || y[i] < 0)
            negative_counter += 1;
    }
    return negative_counter;
}

int getGreaterThanCounter(int* x, const int VALUE, const int LEN) {
    
    int value_counter = 0;
    for (int i = 0; i < LEN; i++) {

        if (x[i] >= VALUE)
            value_counter += 1;
    }
    return value_counter;
}

