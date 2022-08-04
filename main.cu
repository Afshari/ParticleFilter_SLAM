
#include "headers.h"
#include "host_asserts.h"
#include "kernels.cuh"

// Create README File

// For using 'Legayc Tests' please first read the README file
//#define TEST_LEGACY

// Current version of 'RUN_DRAW' is Obsolete
//#define RUN_DRAW

#if defined(TEST_LEGACY)
#include "tests_legacy/test_kernels.h"
#elif defined(RUN_DRAW)
#include "gl_draw.h"
#endif

#include "run_kernels.h"
#include "gtest/gtest.h"


int main(int argc, char* argv[]) {

#if defined(TEST_LEGACY)
    test_main();
    return 0;
#elif defined(RUN_DRAW)
    draw_main();
    return 0;
#endif

    // For Running the Application make sure to run it in 'Release Mode'
    run_main();

    return 0;
}


