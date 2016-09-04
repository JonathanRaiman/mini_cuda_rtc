#include "timer.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace {
    std::unordered_map<std::string, std::atomic<int>> timer_timers;
    std::mutex timer_timers_mutex;
}

Timer::Timer(std::string name, bool autostart) : name(name),
                                                 stopped(false),
                                                 started(false) {
    if (timer_timers.find(name) == timer_timers.end()) {
        std::lock_guard<decltype(timer_timers_mutex)> guard(timer_timers_mutex);
        if (timer_timers.find(name) == timer_timers.end())
            timer_timers[name] = 0;
    }
    if (autostart)
        start();
}

void Timer::start() {
    assert(!started);
    start_time = clock_t::now();
    started = true;
}

void Timer::stop() {
    assert(!stopped);
    timer_timers[name] += std::chrono::duration_cast< std::chrono::microseconds >
                    (clock_t::now() - start_time).count();
    stopped = true;
}

Timer::~Timer() {
    if(!stopped)
        stop();
}

void Timer::report() {
    std::lock_guard<decltype(timer_timers_mutex)> guard(timer_timers_mutex);

    for (auto& kv : timer_timers) {
        std::cout << "\"" << kv.first << "\" => "
                  << std::fixed << std::setw(10) << std::setprecision(10) << std::setfill(' ')
                  << (double) kv.second / 1e6  << "s" << std::endl;
    }

    timer_timers.clear();
}
