// ZERO opening-suite generator.
// Plays N random legal plies from the start position, keeps only positions that
// both a shallow and a deeper ZERO search consider balanced, and writes them as
// an EPD file usable by cutechess-cli / Cute Chess ("-openings file=... format=epd").
//
// Usage: genbook <out.epd> [count=400] [plies=8] [seed=1] [maxAbsCp=35]
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "movegen.h"
#include "piece.h"
#include "position.h"
#include "search.h"

using namespace Zero;

static std::string fen4(const Position& p)
{
    static const char sym[] = " PNBRQKpnbrqk";
    std::string f;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Piece pc = p.piece_on(make_square(file, rank));
            if (pc == EMPTY) { ++empty; continue; }
            if (empty) { f += char('0' + empty); empty = 0; }
            f += sym[pc];
        }
        if (empty) f += char('0' + empty);
        if (rank) f += '/';
    }
    const StateInfo* s = p.state();
    f += p.isWhiteToMove() ? " w " : " b ";
    std::string c;
    if (s->whiteKingSideCastle)  c += 'K';
    if (s->whiteQueenSideCastle) c += 'Q';
    if (s->blackKingSideCastle)  c += 'k';
    if (s->blackQueenSideCastle) c += 'q';
    f += c.empty() ? "-" : c;
    f += ' ';
    if (s->epSquare == SQ_NONE) f += '-';
    else { f += char('a' + file_of(s->epSquare)); f += char('1' + rank_of(s->epSquare)); }
    return f;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::cerr << "usage: genbook out.epd [count] [plies] [seed] [maxAbsCp]\n"; return 1; }
    const std::string outPath = argv[1];
    const int target  = argc > 2 ? std::atoi(argv[2]) : 400;
    const int plies   = argc > 3 ? std::atoi(argv[3]) : 8;
    const unsigned seed = argc > 4 ? unsigned(std::atoi(argv[4])) : 1u;
    const int maxCp   = argc > 5 ? std::atoi(argv[5]) : 35;

    Position::init();
    Position pos;
    Search::Worker worker(pos);
    std::mt19937 rng(seed);
    std::set<std::string> seen;
    std::ofstream out(outPath);
    int kept = 0, tried = 0;

    while (kept < target && tried < 200000) {
        ++tried;
        pos = Position();
        std::deque<StateInfo> states;
        bool ok = true;
        int captures = 0;
        for (int i = 0; i < plies && ok; ++i) {
            auto moves = generateLegalMoves(pos, pos.isWhiteToMove());
            if (moves.empty()) { ok = false; break; }
            // Prefer sensible developing moves: no king/rook moves, no a/h-pawn
            // pushes, no early queen sorties. Falls back to any legal move.
            std::vector<Move> sensible;
            for (const Move& cand : moves) {
                const Piece mover = pos.piece_on(cand.from_sq());
                const PieceType pt = type_of(mover);
                if (pt == KING || pt == ROOK) continue;
                if (pt == QUEEN && i < 6) continue;
                if (pt == PAWN && (file_of(cand.from_sq()) == 0 || file_of(cand.from_sq()) == 7)) continue;
                sensible.push_back(cand);
            }
            const Move m = sensible.empty() ? moves[rng() % moves.size()]
                                            : sensible[rng() % sensible.size()];
            if (pos.piece_on(m.to_sq()) != EMPTY || m.isEnPassant()) ++captures;
            states.emplace_back();
            pos.doMove(m, states.back());
        }
        if (!ok || captures > 2) continue;
        if (pos.isKingInCheck(pos.isWhiteToMove())) continue;
        {
            const StateInfo* st = pos.state();
            if (!(st->whiteKingSideCastle && st->whiteQueenSideCastle
                  && st->blackKingSideCastle && st->blackQueenSideCastle)) continue;
        }
        if (generateLegalMoves(pos, pos.isWhiteToMove()).empty()) continue;

        const std::string f = fen4(pos);
        if (seen.count(f)) continue;

        // Screening: shallow search first (cheap), then a deeper confirmation.
        const auto shallow = worker.searchRoot(4);
        if (!shallow.completed || std::abs(shallow.score) > maxCp + 25) continue;
        const auto deep = worker.searchRoot(8);
        if (!deep.completed || std::abs(deep.score) > maxCp) continue;
        if (std::abs(deep.score - shallow.score) > 45) continue;   // unstable / tactical

        seen.insert(f);
        out << f << '\n';
        ++kept;
        if (kept % 50 == 0) std::cerr << "kept " << kept << " / tried " << tried << '\n';
    }
    std::cerr << "done: kept " << kept << " of " << tried << " candidates -> " << outPath << '\n';
    return kept > 0 ? 0 : 2;
}
