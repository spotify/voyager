#pragma once

#include <mutex>
#include <condition_variable>


namespace voyager {

class ReadWriteLock {
public:
    ReadWriteLock() : readers(0), writers(0), active_writers(0) {}

    void lock_read() {
        std::unique_lock<std::mutex> lock(mutex);
        while (writers > 0 || active_writers > 0) {
            cond_reader.wait(lock);
        }
        ++readers;
    }

    void unlock_read() {
        std::unique_lock<std::mutex> lock(mutex);
        --readers;
        if (readers == 0 && active_writers > 0) {
            cond_writer.notify_one();
        }
    }

    void lock_write() {
        std::unique_lock<std::mutex> lock(mutex);
        ++writers;
        while (readers > 0 || active_writers > 0) {
            cond_writer.wait(lock);
        }
        --writers;
        ++active_writers;
    }

    void unlock_write() {
        std::unique_lock<std::mutex> lock(mutex);
        --active_writers;
        if (writers > 0) {
            cond_writer.notify_one();
        } else {
            cond_reader.notify_all();
        }
    }

private:
    std::mutex mutex;
    std::condition_variable cond_reader;
    std::condition_variable cond_writer;
    int readers;
    int writers;
    int active_writers;
};

} // namespace voyager