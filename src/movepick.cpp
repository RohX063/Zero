#include "movepick.h"
#include "evaluation.h"
#include "position.h"

#include <algorithm>
#include <limits>

namespace {

int scoreMove(const Position& position, const Move& move)
{
    const Piece attacker = position.piece_on(move.from_sq());
    Piece victim = position.piece_on(move.to_sq());

    if (move.isEnPassant())
        victim = make_piece(position.isWhiteToMove() ? Zero::BLACK : Zero::WHITE, Zero::PAWN);

    if (victim != EMPTY)
        return 100000 + pieceValue(victim) * 10 - pieceValue(attacker);

    return 0;
}

} // namespace

MovePicker::MovePicker(const Position& position, const Move* moves, std::size_t count,
                           Move ttMove, Move counterMove, Move killer1, Move killer2)
    : position_(position)
{
    count_ = std::min(count, CAPACITY);
    for (std::size_t i = 0; i < count_; ++i) {
        moves_[i] = {moves[i], scoreMove(position_, moves[i])};

        if (moves[i] == ttMove)
            moves_[i].score = std::numeric_limits<int>::max();
        else if (moves[i] == counterMove && position_.piece_on(moves[i].to_sq()) == EMPTY
                 && !moves[i].isPromotion() && !moves[i].isEnPassant())
            moves_[i].score = 40000;
        else if (moves[i] == killer1 && position_.piece_on(moves[i].to_sq()) == EMPTY
                 && !moves[i].isPromotion() && !moves[i].isEnPassant())
            moves_[i].score = 30000;
        else if (moves[i] == killer2 && position_.piece_on(moves[i].to_sq()) == EMPTY
                 && !moves[i].isPromotion() && !moves[i].isEnPassant())
            moves_[i].score = 29900;
    }

    std::sort(moves_.begin(), moves_.begin() + static_cast<std::ptrdiff_t>(count_),
              [](const auto& a, const auto& b) { return a.score > b.score; });
}

Move MovePicker::next_move()
{
    if (cursor_ >= count_)
        return Move::none();
    return moves_[cursor_++].move;
}
