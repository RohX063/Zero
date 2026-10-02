#include "history.h"

#include <algorithm>
#include <cstdlib>

namespace Zero::Search {

namespace {

template<typename HistoryArray>
void clear2D(HistoryArray& array)
{
    for (auto& row : array)
        row.fill(0);
}

template<typename Table>
void clearNested4(Table& table)
{
    for (auto& a : table)
        for (auto& b : a)
            for (auto& c : b)
                c.fill(0);
}

int updateHistoryValue(HistoryValue current, int bonus, int maxMagnitude)
{
    bonus = std::clamp(bonus, -maxMagnitude, maxMagnitude);

    // Gravity update:
    //
    // new = old + bonus - old * |bonus| / MAX
    //
    // This learns aggressively when the entry is near zero and progressively
    // resists saturation as evidence accumulates.
    const int value = static_cast<int>(current);
    const int absBonus = std::abs(bonus);
    const int updated = value + bonus - (value * absBonus) / maxMagnitude;

    return std::clamp(updated, -maxMagnitude, maxMagnitude);
}

} // namespace

void ButterflyHistory::clear()
{
    for (auto& color : table_)
        clear2D(color);
}

int ButterflyHistory::probe(Color color, Square from, Square to) const
{
    if (color >= COLOR_NB || from >= SQUARE_NB || to >= SQUARE_NB)
        return 0;

    return table_[color][from][to];
}

void ButterflyHistory::update(Color color, Square from, Square to, int bonus)
{
    if (color >= COLOR_NB || from >= SQUARE_NB || to >= SQUARE_NB)
        return;

    table_[color][from][to] = static_cast<HistoryValue>(
        updateHistoryValue(table_[color][from][to], bonus, ButterflyHistory::MAX_VALUE));
}

void PieceToHistory::clear()
{
    for (auto& piece : table_)
        clear2D(piece);
}

int PieceToHistory::probe(Piece piece, Square from, Square to) const
{
    if (piece >= PIECE_NB || from >= SQUARE_NB || to >= SQUARE_NB)
        return 0;

    return table_[piece][from][to];
}

void PieceToHistory::update(Piece piece, Square from, Square to, int bonus)
{
    if (piece >= PIECE_NB || from >= SQUARE_NB || to >= SQUARE_NB)
        return;

    table_[piece][from][to] = static_cast<HistoryValue>(
        updateHistoryValue(table_[piece][from][to], bonus, ButterflyHistory::MAX_VALUE));
}

void CaptureHistory::clear()
{
    for (auto& attacker : table_)
        for (auto& victim : attacker)
            victim.fill(0);
}

int CaptureHistory::probe(Piece attacker, Piece victim, Square to) const
{
    if (attacker >= PIECE_NB || victim >= PIECE_NB || to >= SQUARE_NB)
        return 0;

    return table_[attacker][victim][to];
}

void CaptureHistory::update(Piece attacker, Piece victim, Square to, int bonus)
{
    if (attacker >= PIECE_NB || victim >= PIECE_NB || to >= SQUARE_NB)
        return;

    table_[attacker][victim][to] = static_cast<HistoryValue>(
        updateHistoryValue(table_[attacker][victim][to],
                           bonus,
                           ButterflyHistory::MAX_VALUE));
}

void ContinuationHistory::clear()
{
    clearNested4(table_);
}

int ContinuationHistory::probe(Piece previousPiece, Square previousTo,
                               Piece currentPiece, Square currentTo) const
{
    if (previousPiece >= PIECE_NB || previousTo >= SQUARE_NB
        || currentPiece >= PIECE_NB || currentTo >= SQUARE_NB)
        return 0;

    return table_[previousPiece][previousTo][currentPiece][currentTo];
}

void ContinuationHistory::update(Piece previousPiece, Square previousTo,
                                 Piece currentPiece, Square currentTo, int bonus)
{
    if (previousPiece >= PIECE_NB || previousTo >= SQUARE_NB
        || currentPiece >= PIECE_NB || currentTo >= SQUARE_NB)
        return;

    table_[previousPiece][previousTo][currentPiece][currentTo] =
        static_cast<HistoryValue>(
            updateHistoryValue(
                table_[previousPiece][previousTo][currentPiece][currentTo],
                bonus,
                ButterflyHistory::MAX_VALUE));
}

void HistoryTables::clear()
{
    butterfly_.clear();
    pieceTo_.clear();
    capture_.clear();
    continuation_.clear();
}

int HistoryTables::quietScore(Color color,
                              Piece piece,
                              Square from,
                              Square to,
                              Piece previousPiece,
                              Square previousTo) const
{
    int score = butterfly_.probe(color, from, to)
              + 2 * pieceTo_.probe(piece, from, to);

    if (previousPiece != EMPTY && previousTo < SQUARE_NB)
        score += 3 * continuation_.probe(previousPiece, previousTo, piece, to);

    // Keep history below the tactical / killer score bands used by the
    // current MovePicker. The history still has a wide enough corridor to
    // significantly reorder quiets.
    return std::clamp(score, -8000, 8000);
}

int HistoryTables::captureScore(Piece attacker, Piece victim, Square to) const
{
    return std::clamp(capture_.probe(attacker, victim, to),
                      -4000, 4000);
}

void HistoryTables::updateQuiet(Color color,
                                Piece piece,
                                Square from,
                                Square to,
                                Piece previousPiece,
                                Square previousTo,
                                int bonus)
{
    butterfly_.update(color, from, to, bonus);
    pieceTo_.update(piece, from, to, bonus);

    if (previousPiece != EMPTY && previousTo < SQUARE_NB)
        continuation_.update(previousPiece, previousTo, piece, to, bonus);
}

void HistoryTables::updateCapture(Piece attacker,
                                  Piece victim,
                                  Square to,
                                  int bonus)
{
    capture_.update(attacker, victim, to, bonus);
}

} // namespace Zero::Search
