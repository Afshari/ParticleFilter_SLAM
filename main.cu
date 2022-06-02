
#include "headers.h"
#include "host_asserts.h"
#include "kernels.cuh"

#define RUN_TESTS
//#define RUN_DRAW

#if defined(RUN_TESTS)
#include "test_kernels.h"
#elif defined(RUN_DRAW)
#include "gl_draw.h"
#else
#include "run_kernels.h"
#endif


int main() {

#if defined(RUN_TESTS)
    test_main();
#elif defined(RUN_DRAW)
    draw_main();
#else
    run_main();
#endif

    return 0;
}


