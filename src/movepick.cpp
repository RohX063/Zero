#include "movepick.h"

#include "position.h"

#include <algorithm>
#include <limits>
#include "evaluation/material.h"

namespace {

Piece capturedPiece(const Position& position, const Move& move)
{
    if (move.isEnPassant())
        return make_piece(position.isWhiteToMove() ? Zero::BLACK : Zero::WHITE, Zero::PAWN);

    return position.piece_on(move.to_sq());
}

bool isQuietMove(const Move& move, Piece victim)
{
    return victim == EMPTY && !move.isPromotion() && !move.isEnPassant();
}

int scoreMove(const Position& position,
              const Move& move,
              const Zero::Search::HistoryTables* history,
              Zero::Color side,
              Piece previousPiece,
              Zero::Square previousTo)
{
    const Piece attacker = position.piece_on(move.from_sq());
    const Piece victim = capturedPiece(position, move);

    if (victim != EMPTY) {
        int score = 100000 + pieceValue(victim) * 10 - pieceValue(attacker);

        if (history)
            score += history->captureScore(attacker, victim, move.to_sq());

        return score;
    }

    if (move.isPromotion())
        return 90000 + pieceValue(move.promotionPiece);

    if (history) {
        const int historyScore = history->quietScore(
            side, attacker, move.from_sq(), move.to_sq(), previousPiece, previousTo);

        // Quiet-history score occupies a distinct band below tactical moves,
        // while still leaving enough dynamic range to materially reorder quiets.
        return 12000 + historyScore / 2;
    }

    return 0;
}

} // namespace

MovePicker::MovePicker(const Position& position, const Move* moves, std::size_t count,
                       Move ttMove, Move counterMove, Move killer1, Move killer2,
                       const Zero::Search::HistoryTables* history,
                       Zero::Color side,
                       Piece previousPiece,
                       Zero::Square previousTo)
    : position_(position)
{
    count_ = std::min(count, CAPACITY);

    for (std::size_t i = 0; i < count_; ++i) {
        moves_[i] = {
            moves[i],
            scoreMove(position_, moves[i], history, side, previousPiece, previousTo)
        };

        const Piece victim = capturedPiece(position_, moves[i]);
        const bool quiet = isQuietMove(moves[i], victim);

        if (moves[i] == ttMove)
            moves_[i].score = std::numeric_limits<int>::max();
        else if (moves[i] == counterMove && quiet)
            moves_[i].score = 40000;
        else if (moves[i] == killer1 && quiet)
            moves_[i].score = 30000;
        else if (moves[i] == killer2 && quiet)
            moves_[i].score = 29900;
    }

    std::sort(moves_.begin(),
              moves_.begin() + static_cast<std::ptrdiff_t>(count_),
              [](const auto& a, const auto& b) { return a.score > b.score; });
}

Move MovePicker::next_move()
{
    if (cursor_ >= count_)
        return Move::none();

    return moves_[cursor_++].move;
}
