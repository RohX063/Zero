#ifndef ZERO_SEARCH_PARAMS_H
#define ZERO_SEARCH_PARAMS_H

#include <array>
#include <cstddef>
#include <cstring>

namespace Zero::Search {

// Every selective-search constant that used to be hard-coded lives here and is
// exposed as a UCI spin option, so ZERO's own SPSA/SPRT tooling owns the values.
//
// Defaults are ZERO-native starting points chosen for ZERO's evaluation scale
// (pawn = 100 cp). They are NOT copied from any other engine and are expected
// to move once the tuner has run. Provenance: see docs/PARAMETER_PROVENANCE.md.
struct SearchParams {
    // --- Null-move pruning -------------------------------------------------
    // Prune when staticEval >= beta + nmpMarginBase - nmpMarginPerDepth * depth.
    int nmpMinDepth        = 4;     // must stay >= 3 (static eval is needed)
    int nmpMarginBase      = 110;
    int nmpMarginPerDepth  = 10;
    // Reduction R = nmpReductionBase + depth / nmpDepthDivisor
    //             + min(evalGap / nmpEvalDivisor, nmpEvalCap)
    int nmpReductionBase   = 3;
    int nmpDepthDivisor    = 4;
    int nmpEvalDivisor     = 300;
    int nmpEvalCap         = 3;
    // Deep cutoffs are re-checked with null moves disabled for a window equal
    // to nmpVerifyPercent % of the null-search depth.
    int nmpVerifyDepth     = 10;
    int nmpVerifyPercent   = 65;

    // --- Hindsight depth correction ("mistake punisher") -------------------
    int hindsightRecoverReduction = 3;   // prior reduction needed to give a ply back
    int hindsightTrimReduction    = 2;   // prior reduction needed to take a ply away
    int hindsightTrimEvalSum      = 150; // static-eval sum above which a ply is trimmed

    // --- Extensions ----------------------------------------------------------
    int checkExtension       = 1;    // 0/1
    int promotionExtension   = 1;    // 0/1
    // Extensions are only granted while ply < max(8, rootDepth * pct / 100):
    // prevents extension chains from exploding the tree.
    int extensionPlyLimitPct = 200;

    // --- History -------------------------------------------------------------
    // bonus(depth) = min(historyBonusMax, historyBonusBase + historyBonusPerDepth * depth)
    int historyBonusBase     = 0;
    int historyBonusPerDepth = 110;
    int historyBonusMax      = 1400;
    int historyMalusPercent  = 60;   // malus = bonus * pct / 100 for non-best tried quiets

    // --- Time management -----------------------------------------------------
    int moveOverheadMs       = 20;
};

struct ParamDef {
    const char* name;
    int SearchParams::* field;
    int lo;
    int hi;
};

inline const std::array<ParamDef, 18>& searchParamTable()
{
    static const std::array<ParamDef, 18> table = {{
        {"NMPMinDepth",              &SearchParams::nmpMinDepth,              3,   12},
        {"NMPMarginBase",            &SearchParams::nmpMarginBase,          -200,  600},
        {"NMPMarginPerDepth",        &SearchParams::nmpMarginPerDepth,         0,   40},
        {"NMPReductionBase",         &SearchParams::nmpReductionBase,          1,    6},
        {"NMPDepthDivisor",          &SearchParams::nmpDepthDivisor,           2,    8},
        {"NMPEvalDivisor",           &SearchParams::nmpEvalDivisor,           64,  800},
        {"NMPEvalCap",               &SearchParams::nmpEvalCap,                0,    6},
        {"NMPVerifyDepth",           &SearchParams::nmpVerifyDepth,            6,   30},
        {"NMPVerifyPercent",         &SearchParams::nmpVerifyPercent,         25,  100},
        {"HindsightRecoverReduction",&SearchParams::hindsightRecoverReduction,  2,    8},
        {"HindsightTrimReduction",   &SearchParams::hindsightTrimReduction,     1,    6},
        {"HindsightTrimEvalSum",     &SearchParams::hindsightTrimEvalSum,      0,  600},
        {"CheckExtension",           &SearchParams::checkExtension,            0,    1},
        {"PromotionExtension",       &SearchParams::promotionExtension,        0,    1},
        {"ExtensionPlyLimitPct",     &SearchParams::extensionPlyLimitPct,    100,  400},
        {"HistoryBonusPerDepth",     &SearchParams::historyBonusPerDepth,      8,  400},
        {"HistoryBonusMax",          &SearchParams::historyBonusMax,         200, 4000},
        {"HistoryMalusPercent",      &SearchParams::historyMalusPercent,       0,  150},
    }};
    return table;
}

} // namespace Zero::Search

#endif
