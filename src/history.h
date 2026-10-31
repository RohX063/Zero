#ifndef ZERO_HISTORY_H
#define ZERO_HISTORY_H

#include <array>
#include <cstdint>

#include "move.h"
#include "piece.h"
#include "types.h"

namespace Zero::Search {

// History values are bounded signed statistics. Positive values mean that a
// move has repeatedly produced useful search results in the indexed context;
// negative values mean that it has repeatedly failed in that context.
using HistoryValue = std::int16_t;

class ButterflyHistory {
public:
    static constexpr int MAX_VALUE = 32767;
    static constexpr int MIN_VALUE = -32767;

    ButterflyHistory() { clear(); }

    void clear();
    int probe(Color color, Square from, Square to) const;
    void update(Color color, Square from, Square to, int bonus);

private:
    std::array<std::array<std::array<HistoryValue, SQUARE_NB>, SQUARE_NB>, COLOR_NB> table_{};
};

class PieceToHistory {
public:
    PieceToHistory() { clear(); }

    void clear();
    int probe(Piece piece, Square from, Square to) const;
    void update(Piece piece, Square from, Square to, int bonus);

private:
    std::array<std::array<std::array<HistoryValue, SQUARE_NB>, SQUARE_NB>, PIECE_NB> table_{};
};

class CaptureHistory {
public:
    CaptureHistory() { clear(); }

    void clear();
    int probe(Piece attacker, Piece victim, Square to) const;
    void update(Piece attacker, Piece victim, Square to, int bonus);

private:
    std::array<
        std::array<std::array<HistoryValue, SQUARE_NB>, PIECE_NB>,
        PIECE_NB
    > table_{};
};

class ContinuationHistory {
public:
    ContinuationHistory() { clear(); }

    void clear();
    int probe(Piece previousPiece, Square previousTo,
              Piece currentPiece, Square currentTo) const;
    void update(Piece previousPiece, Square previousTo,
                Piece currentPiece, Square currentTo, int bonus);

private:
    // Previous move context -> current move context.
    //
    // PIECE_NB x SQUARE_NB x PIECE_NB x SQUARE_NB =
    // 13 x 64 x 13 x 64 signed 16-bit entries.
    std::array<
        std::array<
            std::array<
                std::array<HistoryValue, SQUARE_NB>,
                PIECE_NB
            >,
            SQUARE_NB
        >,
        PIECE_NB
    > table_{};
};

class HistoryTables {
public:
    void clear();

    // Composite quiet-move history used by MovePicker.
    int quietScore(Color color,
                   Piece piece,
                   Square from,
                   Square to,
                   Piece previousPiece = EMPTY,
                   Square previousTo = SQ_NONE) const;

    // Capture history is indexed separately because the same destination can
    // have very different search behavior depending on the captured piece.
    int captureScore(Piece attacker, Piece victim, Square to) const;

    // Update all relevant quiet histories with a common learning signal.
    void updateQuiet(Color color,
                     Piece piece,
                     Square from,
                     Square to,
                     Piece previousPiece,
                     Square previousTo,
                     int bonus);

    void updateCapture(Piece attacker,
                       Piece victim,
                       Square to,
                       int bonus);

    // Direct accessors are intentionally small and read-only so regression
    // tests can inspect the learning state without exposing backing storage.
    int butterfly(Color color, Square from, Square to) const {
        return butterfly_.probe(color, from, to);
    }

    int pieceTo(Piece piece, Square from, Square to) const {
        return pieceTo_.probe(piece, from, to);
    }

    int capture(Piece attacker, Piece victim, Square to) const {
        return capture_.probe(attacker, victim, to);
    }

    int continuation(Piece previousPiece, Square previousTo,
                     Piece currentPiece, Square currentTo) const {
        return continuation_.probe(previousPiece, previousTo, currentPiece, currentTo);
    }

private:
    ButterflyHistory butterfly_{};
    PieceToHistory pieceTo_{};
    CaptureHistory capture_{};
    ContinuationHistory continuation_{};
};

} // namespace Zero::Search

#endif
