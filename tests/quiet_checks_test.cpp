// Layer-0 equivalence test: the bitboard quiet-check generator must return EXACTLY the same moves, in
// the same order, as a make/undo oracle on a large set of positions.
//
// Oracle note: the original R3.4 generator passed a Color where Position::isKingInCheck expects a bool, so it
// asked about the wrong king and always returned an empty list. This oracle uses the bool correctly
// (mover's king for legality, enemy king for the check test), i.e. the INTENDED behaviour.
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "movegen.h"
#include "position.h"

using namespace Zero;

static void referenceQuietChecks(Position& position, bool whiteToMove, std::vector<Move>& out)
{
    out.clear();
    MoveList<MAX_MOVES> pseudo;
    generateAllMoves(position, whiteToMove, pseudo);
    for (const Move& move : pseudo) {
        if (move.isPromotion() || move.isEnPassant() || position.piece_on(move.to_sq()) != EMPTY)
            continue;
        StateInfo next;
        position.doMove(move, next);
        const bool legal = !position.isKingInCheck(whiteToMove);        // mover's own king
        const bool givesCheck = position.isKingInCheck(!whiteToMove);   // enemy king
        if (legal && givesCheck)
            out.push_back(move);
        position.undoMove(move);
    }
}

static bool sameMove(const Move& a, const Move& b) { return a.from == b.from && a.to == b.to && a.flags == b.flags; }

static long compare(Position& position, int& failures, long& checksSeen)
{
    std::vector<Move> ref;
    referenceQuietChecks(position, position.isWhiteToMove(), ref);
    TacticalMoveList now;
    generateQuietChecks(position, position.isWhiteToMove(), now);
    checksSeen += static_cast<long>(ref.size());
    bool ok = ref.size() == now.count;
    for (std::size_t i = 0; ok && i < ref.size(); ++i)
        ok = sameMove(ref[i], now.moves[i]);
    if (!ok) {
        if (failures < 5)
            std::cerr << "MISMATCH ref=" << ref.size() << " new=" << now.count << "\n";
        ++failures;
    }
    return 1;
}

int main(int argc, char** argv)
{
    Position::init();
    const int games = argc > 1 ? std::atoi(argv[1]) : 120;
    std::mt19937 rng(20261012);
    long positions = 0;
    long checksSeen = 0;
    int failures = 0;

    const char* fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1",
        "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1",
    };
    for (const char* fen : fens) {
        Position p;
        if (p.loadFEN(fen))
            positions += compare(p, failures, checksSeen);
    }

    for (int g = 0; g < games; ++g) {
        Position pos;
        std::deque<StateInfo> states;
        for (int ply = 0; ply < 160; ++ply) {
            positions += compare(pos, failures, checksSeen);
            auto moves = generateLegalMoves(pos, pos.isWhiteToMove());
            if (moves.empty())
                break;
            states.emplace_back();
            pos.doMove(moves[rng() % moves.size()], states.back());
        }
    }

    std::cout << "positions " << positions << " quiet checks compared " << checksSeen
              << " mismatches " << failures << "\n";
    // The test is meaningless if the oracle never produced a check.
    if (checksSeen == 0)
        return 2;
    return failures == 0 ? 0 : 1;
}
