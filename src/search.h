#ifndef ZERO_SEARCH_H
#define ZERO_SEARCH_H

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "countermove.h"
#include "history.h"
#include "killer.h"
#include "move.h"
#include "movegen.h"
#include "tt.h"
#include "types.h"
#include "search_params.h"
#include "search_stack.h"
#include "search_types.h"
#include "search_time.h"

class Position;

namespace Zero::Search {

class Worker {
public:
    explicit Worker(Position& position);

    SearchResult searchRoot(Depth depth);
    SearchResult iterativeDeepening(Depth maxDepth);
    SearchResult iterativeDeepening(const Limits& limits);

    uint64_t nodes() const { return nodes_; }
    void resetNodes() { nodes_ = 0; }

    void visitNode(int ply = 0) {
        ++nodes_;
        ++clockNodes_;
        if (ply > selDepth_)
            selDepth_ = ply;
    }
    bool shouldStop() const;

    void setHashSize(std::size_t megabytes) { tt_.resize(megabytes); }

    // Correct hash-reset semantics: UCI ucinewgame must clear the TT itself.
    void clearHash() { tt_.clear(); }

    void clearHistory() {
        history_->clear();
        counterMoves_.clear();
    }

    void clearKillers() { killerMoves_.clear(); }

    void setLMRLogScale(int value);
    // Generic UCI-tunable parameter hook (SearchParams + LMRLogScale + Move Overhead).
    bool setParam(const std::string& name, int value);
    const SearchParams& params() const { return params_; }

    void setLMRBase(int value);
    void setLMRDepthCoeff(int value);
    void setLMRMoveCoeff(int value);
    void setLMRPVAdjust(int value);
    void setLMRCutAdjust(int value);
    void setLMRTTAdjust(int value);
    void setLMRHistoryCoeff(int value);
    void setLMRContinuationCoeff(int value);
    void setLMRImprovingAdjust(int value);
    void setLMRTacticalSafetyAdjust(int value);
    const LMRParameters& lmrParameters() const { return lmrParams_; }

    Move killerMoveFor(Depth ply, std::size_t slot) const;
    Move counterMoveFor(Piece previousPiece, Square previousTo) const;
    std::uint32_t hashfull() const { return tt_.hashfull(); }
    std::size_t hashSizeMb() const { return tt_.megabytes(); }
    std::uint64_t nps() const;
    std::uint64_t elapsedMs() const;
    Depth seldepth() const { return selDepth_; }
    const SearchStats& stats() const { return stats_; }

private:
    void rebuildLMRTable();
    int lmrReduction(Depth depth,
                     int moveCount,
                     int historyScore,
                     int continuationHistoryScore,
                     bool improving,
                     bool pvNode,
                     bool cutNode,
                     bool ttMove,
                     bool tacticalSafety,
                     int alphaBetaWindow) const;

    Value search(Position& position, Stack* ss, Depth depth, Value alpha, Value beta, bool cutNode);
    SearchResult searchRootIteration(Depth depth);

    void beginSearch(const Limits& limits);

    Position& position_;
    TranspositionTable tt_{};
    CounterMoveHistory counterMoves_{};
    KillerMoves killerMoves_{};
    // Continuation history is intentionally heap-owned because the full history
    // subsystem is large (~1.5 MiB). Keeping it inline makes Worker itself
    // large enough to overflow the default Windows thread stack when a Worker
    // is created as an automatic/local object (e.g. in tests).
    std::unique_ptr<HistoryTables> history_;
    std::array<Stack, MAX_PLY + 2> stack_{};
    std::array<std::array<int, MAX_MOVES + 1>, MAX_PLY + 1> lmrTable_{};
    LMRParameters lmrParams_{};
    SearchParams params_{};
    SearchStats stats_{};
    uint64_t nodes_ = 0;
    uint64_t clockNodes_ = 0;

    std::chrono::steady_clock::time_point deadline_{};
    std::chrono::steady_clock::time_point searchStart_{};
    TimeBudget timeBudget_{};
    Move lastBestMove_{};
    Move rootBestMove_{};      // best move of the last completed iteration (ordering anchor)
    Depth rootDepth_ = 0;      // depth of the iteration currently running
    Value lastScore_ = VALUE_DRAW;
    int stableIterations_ = 0;
    Depth selDepth_ = 0;
    mutable bool stopped_ = false;   // sticky: once the deadline passes it stays set
};

}

#endif
