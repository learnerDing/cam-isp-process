//以下是Thread.cpp
#include "Thread.h"

Thread::Thread() {}

Thread::~Thread() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void Thread::start() {
    m_thread = std::thread(&Thread::run, this);
}

void Thread::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}