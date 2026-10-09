#ifndef ZERO_SEARCH_TYPES_H
#define ZERO_SEARCH_TYPES_H

#include <cstdint>

#include "move.h"
#include "types.h"

namespace Zero::Search {

struct LMRParameters {
    // All adjustments are in 1/1024-ply reduction units.
    // Table model: R = base + logScale * ln(depth) * ln(moveCount)
    //                        + depthCoeff * ln(depth) + moveCoeff * ln(moveCount)
    // logScale used to be frozen to a value borrowed from another engine; it is
    // now a first-class tunable (LMRLogScale).
    int base = 820;
    int logScale = 470;
    int depthCoeff = 0;
    int moveCoeff = 0;
    int pvAdjust = -512;
    int cutAdjust = 512;
    int ttAdjust = -1024;
    int historyCoeff = -80;
    int continuationCoeff = 0;
    int improvingAdjust = -256;
    int tacticalSafetyAdjust = 0;
};

struct SearchStats {
    std::uint64_t pvsReSearches = 0;
    std::uint64_t lmrReductions = 0;
    std::uint64_t lmrResearches = 0;
    std::uint64_t nullMoveSearches = 0;
    std::uint64_t nullMoveCutoffs = 0;
    std::uint64_t nullMoveVerifications = 0;
    std::uint64_t checkExtensions = 0;
    std::uint64_t promotionExtensions = 0;
    std::uint64_t quietChecksInQSearch = 0;
};

struct Limits {
    Depth depth = 0;
    int movetimeMs = 0;
    int sideTimeMs = 0;
    int incrementMs = 0;
    int movesToGo = 0;
    int moveOverheadMs = 20;
};

struct SearchResult {
    Move bestMove{};
    Value score = -VALUE_INFINITE;
    Depth depth = 0;
    bool completed = false;
};

} // namespace Zero::Search

#endif
