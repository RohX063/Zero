#ifndef ZERO_SEARCH_TIME_H
#define ZERO_SEARCH_TIME_H

#include <chrono>
#include "search_types.h"

namespace Zero::Search {

struct TimeBudget {
    int optimumMs = 0;
    int maximumMs = 0;
};

// Allocate a conservative thinking budget from the current clock state.
// This is intentionally engine-agnostic: the search decides whether the
// current position deserves to approach the maximum budget.
TimeBudget computeTimeBudget(const Limits& limits);

// Decide whether iterative deepening should enter another iteration after a
// completed root search. Best-move instability and score volatility are used
// as signals that another iteration is worth the clock cost.
bool shouldContinueAfterIteration(const TimeBudget& budget,
                                  int elapsedMs,
                                  const Move& previousBestMove,
                                  const Move& currentBestMove,
                                  Value previousScore,
                                  Value currentScore,
                                  int stableIterations);

// Update a small root-PV stability counter without introducing search state.
int updateStableIterations(const Move& previousBestMove,
                           const Move& currentBestMove,
                           Value previousScore,
                           Value currentScore,
                           int stableIterations);

} // namespace Zero::Search

#endif
