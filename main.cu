
#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "test_kernels.h"

#define RUN_TESTS

// ✓


int main() {

#ifdef RUN_TESTS
    test_main();
#endif

    return 0;
}


