#ifndef ZERO_SEARCH_H
#define ZERO_SEARCH_H

#include <array>
#include <chrono>
#include <cstdint>

#include "countermove.h"
#include "history.h"
#include "killer.h"
#include "move.h"
#include "tt.h"
#include "types.h"

class Position;

namespace Zero::Search {

struct Stack {
    Move currentMove{};
    Move excludedMove{};
    Piece movedPiece = EMPTY;
    Value staticEval = 0;
    int ply = 0;
    int moveCount = 0;
    bool inCheck = false;
    bool canNullMove = true;
};

struct SearchStats {
    std::uint64_t pvsReSearches = 0;
    std::uint64_t lmrReductions = 0;
    std::uint64_t nullMoveSearches = 0;
    std::uint64_t nullMoveCutoffs = 0;
    std::uint64_t checkExtensions = 0;
    std::uint64_t promotionExtensions = 0;
};

struct Limits {
    Depth depth = 0;
    int movetimeMs = 0;
    int sideTimeMs = 0;
    int incrementMs = 0;
    int movesToGo = 0;
};

struct SearchResult {
    Move bestMove{};
    Value score = -VALUE_INFINITE;
    Depth depth = 0;
    bool completed = false;
};

class Worker {
public:
    explicit Worker(Position& position);

    SearchResult searchRoot(Depth depth);
    SearchResult iterativeDeepening(Depth maxDepth);
    SearchResult iterativeDeepening(const Limits& limits);

    uint64_t nodes() const { return nodes_; }
    void resetNodes() { nodes_ = 0; }

    void visitNode() { ++nodes_; ++clockNodes_; }
    bool shouldStop() const;

    void setHashSize(std::size_t megabytes) { tt_.resize(megabytes); }
    void clearHash() { tt_.clear(); }
    void clearHistory() {
        history_.clear();
        counterMoves_.clear();
    }
    void clearKillers() { killerMoves_.clear(); }

    Move killerMoveFor(Depth ply, std::size_t slot) const;
    Move counterMoveFor(Piece previousPiece, Square previousTo) const;
    std::uint32_t hashfull() const { return tt_.hashfull(); }
    std::size_t hashSizeMb() const { return tt_.megabytes(); }
    const SearchStats& stats() const { return stats_; }

private:
    Value search(Position& position, Stack* ss, Depth depth, Value alpha, Value beta);
    SearchResult searchRootIteration(Depth depth);

    void beginSearch(const Limits& limits);
    int computeTimeBudgetMs(const Limits& limits) const;

    Position& position_;
    TranspositionTable tt_{};
    CounterMoveHistory counterMoves_{};
    KillerMoves killerMoves_{};
    HistoryTables history_{};
    std::array<Stack, MAX_PLY + 2> stack_{};
    SearchStats stats_{};
    uint64_t nodes_ = 0;
    uint64_t clockNodes_ = 0;

    std::chrono::steady_clock::time_point deadline_{};
};

}

#endif
