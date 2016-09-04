#ifndef RTC_MAKE_MESSAGE_H
#define RTC_MAKE_MESSAGE_H

#include <sstream>

void make_message(std::stringstream* ss);

template<typename Arg, typename... Args>
void make_message(std::stringstream* ss, const Arg& arg, const Args&... args) {
    (*ss) << arg;
    make_message(ss, args...);
}

template<typename... Args>
std::string make_message(const Args&... args) {
    std::stringstream ss;
    make_message(&ss, args...);
    return ss.str();
}

#endif
