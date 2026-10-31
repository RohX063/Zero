#include "thread.h"
#include "position.h"

namespace Zero {
WorkerThread::WorkerThread(Position& position) : worker_(position) {}
ThreadPool::ThreadPool(Position& position) : mainThread_(std::make_unique<WorkerThread>(position)) {}
}
