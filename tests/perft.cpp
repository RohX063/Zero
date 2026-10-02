#include "movegen.h"
#include "position.h"
#include <cstdint>
#include <iostream>

namespace {
uint64_t perft(Position& position, int depth) {
    if (depth == 0) return 1;
    uint64_t nodes = 0;
    for (const Move& move : generateLegalMoves(position, position.isWhiteToMove())) {
        StateInfo state;
        position.doMove(move, state);
        nodes += perft(position, depth - 1);
        position.undoMove(move);
    }
    return nodes;
}

struct Case { const char* name; const char* fen; int depth; uint64_t expected; };

bool runCase(const Case& c) {
    Position position;
    if (!position.loadFEN(c.fen)) { std::cerr << c.name << ": FEN failed\n"; return false; }
    const uint64_t got = perft(position, c.depth);
    std::cout << c.name << ": " << got << " (expected " << c.expected << ")\n";
    return got == c.expected;
}
}

int main() {
    Position::init();
    constexpr Case cases[] = {
        {"start d5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609ULL},
        {"kiwipete d4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ULL},
        {"position3 d4", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238ULL},
        {"position4 d3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467ULL},
        {"position5 d3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379ULL},
        {"position6 d3", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890ULL},
    };
    bool ok = true;
    for (const auto& c : cases) ok &= runCase(c);
    return ok ? 0 : 1;
}
