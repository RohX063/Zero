#include "movepick.h"

#include "position.h"
#include "evaluation/material.h"

#include <limits>

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
        return 12000 + historyScore / 2;
    }

    return 0;
}

bool isCaptureOrPromotion(const MovePicker::ScoredMove& scored,
                          const Position& position,
                          const Move&, const Move&, const Move&, const Move&)
{
    const Move& move = scored.move;
    const Piece victim = capturedPiece(position, move);
    return victim != EMPTY || move.isPromotion() || move.isEnPassant();
}

bool isSpecialQuiet(const MovePicker::ScoredMove& scored,
                    const Position& position,
                    const Move&,
                    const Move& counterMove,
                    const Move& killer1,
                    const Move& killer2)
{
    const Move& move = scored.move;
    const Piece victim = capturedPiece(position, move);
    if (!isQuietMove(move, victim))
        return false;

    return move == counterMove || move == killer1 || move == killer2;
}

bool isQuiet(const MovePicker::ScoredMove& scored,
             const Position& position,
             const Move&, const Move&, const Move&, const Move&)
{
    const Piece victim = capturedPiece(position, scored.move);
    return isQuietMove(scored.move, victim);
}

} // namespace

MovePicker::MovePicker(const Position& position, const Move* moves, std::size_t count,
                       Move ttMove, Move counterMove, Move killer1, Move killer2,
                       const Zero::Search::HistoryTables* history,
                       Zero::Color side,
                       Piece previousPiece,
                       Zero::Square previousTo)
    : position_(position),
      ttMove_(ttMove),
      counterMove_(counterMove),
      killer1_(killer1),
      killer2_(killer2)
{
    count_ = std::min(count, CAPACITY);

    for (std::size_t i = 0; i < count_; ++i)
        moves_[i] = {
            moves[i],
            scoreMove(position_, moves[i], history, side, previousPiece, previousTo)
        };
}

Move MovePicker::selectBest(
    bool (*predicate)(const ScoredMove&, const Position&, const Move&, const Move&, const Move&, const Move&),
    const Move& ttMove,
    const Move& counterMove,
    const Move& killer1,
    const Move& killer2)
{
    std::size_t best = count_;
    int bestScore = std::numeric_limits<int>::min();

    for (std::size_t i = 0; i < count_; ++i) {
        if (used_[i])
            continue;
        if (!predicate(moves_[i], position_, ttMove, counterMove, killer1, killer2))
            continue;

        if (best == count_ || moves_[i].score > bestScore) {
            best = i;
            bestScore = moves_[i].score;
        }
    }

    if (best == count_)
        return Move::none();

    used_[best] = true;
    return moves_[best].move;
}

Move MovePicker::next_move()
{
    for (;;) {
        switch (stage_) {
            case STAGE_TT: {
                stage_ = STAGE_CAPTURES;
                if (!ttMove_.isOk())
                    continue;

                for (std::size_t i = 0; i < count_; ++i) {
                    if (!used_[i] && moves_[i].move == ttMove_) {
                        used_[i] = true;
                        return moves_[i].move;
                    }
                }
                continue;
            }

            case STAGE_CAPTURES: {
                const Move move = selectBest(isCaptureOrPromotion,
                                             ttMove_, counterMove_, killer1_, killer2_);
                if (move.isOk())
                    return move;
                stage_ = STAGE_SPECIAL_QUIETS;
                continue;
            }

            case STAGE_SPECIAL_QUIETS: {
                const Move move = selectBest(isSpecialQuiet,
                                             ttMove_, counterMove_, killer1_, killer2_);
                if (move.isOk())
                    return move;
                stage_ = STAGE_QUIETS;
                continue;
            }

            case STAGE_QUIETS: {
                const Move move = selectBest(isQuiet,
                                             ttMove_, counterMove_, killer1_, killer2_);
                if (move.isOk())
                    return move;
                stage_ = STAGE_DONE;
                continue;
            }

            case STAGE_DONE:
            default:
                return Move::none();
        }
    }
}
