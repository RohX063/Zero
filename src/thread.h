#ifndef ZERO_THREAD_H
#define ZERO_THREAD_H

#include <memory>
#include "search.h"

class Position;

namespace Zero {

class WorkerThread {
public:
    explicit WorkerThread(Position& position);
    Search::Worker& worker() { return worker_; }
    const Search::Worker& worker() const { return worker_; }
private:
    Search::Worker worker_;
};

class ThreadPool {
public:
    explicit ThreadPool(Position& position);
    WorkerThread& mainThread() { return *mainThread_; }
    const WorkerThread& mainThread() const { return *mainThread_; }
private:
    std::unique_ptr<WorkerThread> mainThread_;
};

}

#endif
