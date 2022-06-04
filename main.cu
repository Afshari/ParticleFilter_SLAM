
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

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>  
namespace fs = std::experimental::filesystem;

int main() {

    //string dir = "data/map/";
    //for (const auto& file : fs::directory_iterator(dir))
    //    std::cout << file.path() << std::endl;

#if defined(RUN_TESTS)
    test_main();
#elif defined(RUN_DRAW)
    draw_main();
#else
    run_main();
#endif

    return 0;
}


