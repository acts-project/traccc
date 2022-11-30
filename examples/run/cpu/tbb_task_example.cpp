/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// TBB include(s).
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

// System include(s).
#include <iostream>
#include <thread>

int main() {

    // Arena to run all the tasks in.
    tbb::task_arena arena;
    // Task group to put tasks in. Allowing us to wait for it explicitly.
    tbb::task_group group;

    // To be able to use "time literals".
    using namespace std::chrono_literals;

    // Execute a few hello world tasks.
    for (std::size_t i = 0; i < 32; ++i) {
        arena.execute([i, &group]() {
            group.run([i]() {
                std::cout << "Hello World from task " << i << " on thread "
                          << tbb::this_task_arena::current_thread_index()
                          << std::endl;
                std::this_thread::sleep_for(50ms);
            });
        });
    }

    // Wait for all threads to finish.
    group.wait();

    // Return gracefully.
    return 0;
}
