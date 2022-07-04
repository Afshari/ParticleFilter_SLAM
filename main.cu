
#include "headers.h"
#include "host_asserts.h"
#include "kernels.cuh"

//#define RUN_TESTS
//#define RUN_DRAW

#if defined(RUN_TESTS)
#include "test_kernels.h"
#elif defined(RUN_DRAW)
#include "gl_draw.h"
#else
#include "run_kernels.h"
#endif

// [ ] - Create a function to generate random numbers
// [ ] - Create random number for 100 hundered each time


int main() {

#if defined(RUN_TESTS)
    test_main();
#elif defined(RUN_DRAW)
    draw_main();
#else
    run_main();

    //vector<float> vec;
    //auto start_random_generator = std::chrono::high_resolution_clock::now();
    //for (int i = 0; i < 3; i++) {
    //    gen_random_numbers(vec);
    //    for (auto i : vec)
    //        std::cout << i << " ";
    //    std::cout << std::endl;
    //}
    //auto stop_random_generator = std::chrono::high_resolution_clock::now();
    //auto duration_random_generator = std::chrono::duration_cast<std::chrono::microseconds>(stop_random_generator - start_random_generator);
    //std::cout << "Time taken by function (Random Generator): " << duration_random_generator.count() << " microseconds" << std::endl;


#endif

    return 0;
}


