#ifndef DALI_UTILS_TIMER_H
#define DALI_UTILS_TIMER_H

#include <atomic>
#include <chrono>
#include <string>

namespace utils {
    class Timer {
        typedef std::chrono::system_clock clock_t;

        std::string name;
        bool stopped;
        bool started;
        std::chrono::time_point<clock_t> start_time;

        public:
            // creates timer and starts measuring time.
            Timer(std::string name, bool autostart=true);
            // destroys timer and stops counting if the timer was not previously stopped.
            ~Timer();

            // explicitly start the timer
            void start();
            // explicitly stop the timer
            void stop();

            static void report();
    };
}  // namespace utils

#endif  // DALI_UTILS_TIMER_H
