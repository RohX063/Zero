#include "search_helpers.h"

#include <algorithm>
#include <cmath>

#include <string>

#include "position.h"

namespace Zero::Search {


bool isQuietMove(const Move& move, Piece victim)
{
    return victim == EMPTY && !move.isPromotion() && !move.isEnPassant();
}

Piece capturedPiece(const Position& position, const Move& move)
{
    if (move.isEnPassant())
        return make_piece(position.isWhiteToMove() ? BLACK : WHITE, PAWN);

    return position.piece_on(move.to_sq());
}

int historyBonus(const SearchParams& params, Depth depth)
{
    const int d = std::max(1, depth);
    return std::clamp(params.historyBonusBase + params.historyBonusPerDepth * d,
                      1, params.historyBonusMax);
}

std::string uciMoveString(const Move& move)
{
    if (!move.isOk())
        return "0000";
    std::string text;
    text += char('a' + file_of(move.from));
    text += char('1' + rank_of(move.from));
    text += char('a' + file_of(move.to));
    text += char('1' + rank_of(move.to));
    if (move.isPromotion()) {
        switch (type_of(move.promotionPiece)) {
            case QUEEN:  text += 'q'; break;
            case ROOK:   text += 'r'; break;
            case BISHOP: text += 'b'; break;
            case KNIGHT: text += 'n'; break;
            default: break;
        }
    }
    return text;
}

std::string uciScoreString(Value score)
{
    // Mate scores are stored as +/-(VALUE_MATE - plies). UCI wants full moves.
    if (score >= VALUE_MATE - MAX_PLY)
        return "mate " + std::to_string((VALUE_MATE - score + 1) / 2);
    if (score <= -VALUE_MATE + MAX_PLY)
        return "mate -" + std::to_string((VALUE_MATE + score + 1) / 2);
    return "cp " + std::to_string(score);
}

int evaluateLMRFormula(const LMRParameters& params, int depth, int moveCount)
{
    const double d = std::log(double(std::max(1, depth)));
    const double m = std::log(double(std::max(1, moveCount)));
    const double raw = double(params.base)
                     + double(params.logScale) * d * m
                     + double(params.depthCoeff) * d
                     + double(params.moveCoeff) * m;
    return static_cast<int>(raw);
}

} // namespace Zero::Search
