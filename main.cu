
#include "headers.h"
#include "host_asserts.h"
#include "kernels.cuh"

//#define RUN_TESTS

#if defined(RUN_TESTS)
#include "test_kernels.h"
#else
#include "run_kernels.h"
#endif


int main() {

#if defined(RUN_TESTS)
    test_main();
#else
    run_main();
#endif

    return 0;
}


