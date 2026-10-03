#include "movegen.h"
#include "bitboard.h"
#include "position.h"
#include "piece.h"

using namespace Zero;

namespace {

constexpr int knightOffsets[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    { 1, -2}, { 1, 2}, { 2, -1}, { 2, 1}
};

constexpr int bishopDirections[4][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

constexpr int rookDirections[4][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

constexpr int queenDirections[8][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

constexpr int kingOffsets[8][2] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
    {0, 1}, {1, -1}, {1, 0}, {1, 1}
};

inline bool onBoard(int row, int col) {
    return row >= 0 && row < 8 && col >= 0 && col < 8;
}

inline Square oldSquare(int row, int col) {
    return make_square(col, 7 - row);
}

template<typename List>
inline void addMove(List& moves, Square from, Square to) {
    Move m;
    m.from = from;
    m.to = to;
    moves.push(m);
}

template<typename List>
void addPromotions(List& moves, Square from, Square to, Color c) {
    constexpr Piece promotionPiecesWhite[] = {
        WHITE_QUEEN, WHITE_ROOK, WHITE_BISHOP, WHITE_KNIGHT
    };
    constexpr Piece promotionPiecesBlack[] = {
        BLACK_QUEEN, BLACK_ROOK, BLACK_BISHOP, BLACK_KNIGHT
    };
    const auto& pieces = c == WHITE ? promotionPiecesWhite : promotionPiecesBlack;
    for (Piece pc : pieces) {
        Move m;
        m.from = from;
        m.to = to;
        m.flags = Move::PROMOTION;
        m.promotionPiece = pc;
        moves.push(m);
    }
}

template<typename List>
void appendSlidingFrom(const Position& pos,
                       List& moves,
                       Color us,
                       Square from,
                       const int dirs[][2],
                       int count) {
    const int row = 7 - rank_of(from);
    const int col = file_of(from);
    const Bitboard occupied = pos.pieces();
    const Bitboard own = pos.pieces(us);
    const Bitboard enemyKing = pos.pieces(~us, KING);

    for (int d = 0; d < count; ++d) {
        int r = row + dirs[d][0];
        int c = col + dirs[d][1];
        while (onBoard(r, c)) {
            const Square to = oldSquare(r, c);
            const Bitboard toBB = square_bb(to);
            if (own & toBB) break;
            if (!(enemyKing & toBB)) addMove(moves, from, to);
            if (occupied & toBB) break;
            r += dirs[d][0];
            c += dirs[d][1];
        }
    }
}

template<typename List>
void appendPawnFrom(const Position& pos, List& moves, Color us, Square from) {
    const int row = 7 - rank_of(from);
    const int col = file_of(from);
    const int forward = us == WHITE ? -1 : 1;
    const int promotionRow = us == WHITE ? 0 : 7;
    const int startRow = us == WHITE ? 6 : 1;

    const int oneRow = row + forward;
    if (onBoard(oneRow, col) && pos.piece_on(oldSquare(oneRow, col)) == EMPTY) {
        const Square one = oldSquare(oneRow, col);
        if (oneRow == promotionRow) addPromotions(moves, from, one, us);
        else addMove(moves, from, one);

        const int twoRow = row + 2 * forward;
        if (row == startRow && onBoard(twoRow, col)
            && pos.piece_on(oldSquare(twoRow, col)) == EMPTY)
            addMove(moves, from, oldSquare(twoRow, col));
    }

    for (int dc : {-1, 1}) {
        const int capRow = row + forward;
        const int capCol = col + dc;
        if (!onBoard(capRow, capCol)) continue;
        const Square to = oldSquare(capRow, capCol);
        const Piece target = pos.piece_on(to);
        if (target != EMPTY && ((us == WHITE && isBlackPiece(target))
                                || (us == BLACK && isWhitePiece(target)))) {
            if (capRow == promotionRow) addPromotions(moves, from, to, us);
            else addMove(moves, from, to);
        }
    }

    const Square ep = pos.ep_square();
    if (ep != SQ_NONE && (Bitboards::PawnAttacks[us][from] & square_bb(ep))) {
        Move m;
        m.from = from;
        m.to = ep;
        m.flags = Move::EN_PASSANT;
        moves.push(m);
    }
}

template<typename List>
void appendKnightFrom(const Position& pos, List& moves, Color us, Square from) {
    const int row = 7 - rank_of(from);
    const int col = file_of(from);
    const Bitboard own = pos.pieces(us);
    const Bitboard enemyKing = pos.pieces(~us, KING);

    for (const auto& delta : knightOffsets) {
        const int r = row + delta[0], c = col + delta[1];
        if (!onBoard(r, c)) continue;
        const Square to = oldSquare(r, c);
        const Bitboard bb = square_bb(to);
        if (own & bb) continue;
        if (enemyKing & bb) continue;
        addMove(moves, from, to);
    }
}

template<typename List>
void appendKingFrom(const Position& pos, List& moves, Color us, Square from) {
    const int row = 7 - rank_of(from);
    const int col = file_of(from);
    const Bitboard own = pos.pieces(us);
    const Bitboard enemyKing = pos.pieces(~us, KING);

    for (const auto& delta : kingOffsets) {
        const int r = row + delta[0], c = col + delta[1];
        if (!onBoard(r, c)) continue;
        const Square to = oldSquare(r, c);
        const Bitboard bb = square_bb(to);
        if (own & bb) continue;
        if (enemyKing & bb) continue;
        addMove(moves, from, to);
    }

    const int rank = us == WHITE ? 0 : 7;
    const Square e = make_square(4, rank);
    if (from != e || pos.isSquareAttacked(e, ~us)) return;

    const Piece rook = us == WHITE ? WHITE_ROOK : BLACK_ROOK;
    if (us == WHITE ? pos.canWhiteCastleKingSide() : pos.canBlackCastleKingSide()) {
        const Square f = make_square(5, rank), g = make_square(6, rank), h = make_square(7, rank);
        if (pos.piece_on(f) == EMPTY && pos.piece_on(g) == EMPTY && pos.piece_on(h) == rook
            && !pos.isSquareAttacked(f, ~us) && !pos.isSquareAttacked(g, ~us)) {
            Move castle;
            castle.from = e; castle.to = g; castle.flags = Move::CASTLE_KING;
            moves.push(castle);
        }
    }

    if (us == WHITE ? pos.canWhiteCastleQueenSide() : pos.canBlackCastleQueenSide()) {
        const Square b = make_square(1, rank), c = make_square(2, rank), d = make_square(3, rank), a = make_square(0, rank);
        if (pos.piece_on(b) == EMPTY && pos.piece_on(c) == EMPTY && pos.piece_on(d) == EMPTY && pos.piece_on(a) == rook
            && !pos.isSquareAttacked(d, ~us) && !pos.isSquareAttacked(c, ~us)) {
            Move castle;
            castle.from = e; castle.to = c; castle.flags = Move::CASTLE_QUEEN;
            moves.push(castle);
        }
    }
}

} // namespace

void generateAllMoves(const Position& position, bool whiteToMove, MoveList<MAX_MOVES>& moves) {
    moves.count = 0;
    const Color us = whiteToMove ? WHITE : BLACK;

    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            const Square from = oldSquare(row, col);
            const Piece piece = position.piece_on(from);
            if (piece == EMPTY || color_of(piece) != us) continue;

            switch (type_of(piece)) {
                case PAWN:   appendPawnFrom(position, moves, us, from); break;
                case KNIGHT: appendKnightFrom(position, moves, us, from); break;
                case BISHOP: appendSlidingFrom(position, moves, us, from, bishopDirections, 4); break;
                case ROOK:   appendSlidingFrom(position, moves, us, from, rookDirections, 4); break;
                case QUEEN:  appendSlidingFrom(position, moves, us, from, queenDirections, 8); break;
                case KING:   appendKingFrom(position, moves, us, from); break;
                default: break;
            }
        }
    }
}

void generateLegalMoves(Position& position, bool whiteToMove, MoveList<MAX_MOVES>& legalMoves) {
    MoveList<MAX_MOVES> pseudo;
    generateAllMoves(position, whiteToMove, pseudo);
    legalMoves.count = 0;

    for (const Move& move : pseudo) {
        StateInfo next;
        position.doMove(move, next);
        if (!position.isKingInCheck(whiteToMove))
            legalMoves.push(move);
        position.undoMove(move);
    }
}

void generateCaptureMoves(Position& position, bool whiteToMove, MoveList<MAX_MOVES>& captures) {
    MoveList<MAX_MOVES> legal;
    generateLegalMoves(position, whiteToMove, legal);
    captures.count = 0;
    for (const Move& move : legal) {
        if (move.isEnPassant() || position.piece_on(move.to) != EMPTY || move.isPromotion())
            captures.push(move);
    }
}

void generateQuietChecks(Position& position,
                         bool whiteToMove,
                         TacticalMoveList& list)
{
    list.count = 0;

    const Color us = whiteToMove ? WHITE : BLACK;

    MoveList<MAX_MOVES> pseudo;
    generateAllMoves(position, whiteToMove, pseudo);

    for (const Move& move : pseudo) {
        // Captures/promotions are already covered by generateTacticalMoves().
        if (move.isPromotion() || move.isEnPassant() || position.piece_on(move.to_sq()) != EMPTY)
            continue;

        StateInfo next;
        position.doMove(move, next);

        const bool legal = !position.isKingInCheck(us);
        const bool givesCheck = position.isKingInCheck(!whiteToMove);

        if (legal && givesCheck)
            list.push(move);

        position.undoMove(move);
    }
}

std::vector<Move> generateAllMoves(const Position& position, bool whiteToMove) {
    MoveList<MAX_MOVES> list;
    generateAllMoves(position, whiteToMove, list);
    return std::vector<Move>(list.begin(), list.end());
}

std::vector<Move> generateLegalMoves(Position& position, bool whiteToMove) {
    MoveList<MAX_MOVES> list;
    generateLegalMoves(position, whiteToMove, list);
    return std::vector<Move>(list.begin(), list.end());
}

std::vector<Move> generateCaptureMoves(Position& position, bool whiteToMove) {
    MoveList<MAX_MOVES> list;
    generateCaptureMoves(position, whiteToMove, list);
    return std::vector<Move>(list.begin(), list.end());
}

namespace {

inline void addTacticalMove(TacticalMoveList& list, Move move) {
    if (list.count < MAX_TACTICAL_MOVES)
        list.moves[list.count++] = move;
}

inline void addTacticalPromotions(TacticalMoveList& list, Square from, Square to, Color us) {
    for (Piece pt : {make_piece(us, QUEEN), make_piece(us, ROOK),
                     make_piece(us, BISHOP), make_piece(us, KNIGHT)}) {
        Move move;
        move.from = from;
        move.to = to;
        move.flags = Move::PROMOTION;
        move.promotionPiece = pt;
        addTacticalMove(list, move);
    }
}
}

TacticalMoveList generateTacticalMoves(const Position& position, bool whiteToMove) {
    const Color us = whiteToMove ? WHITE : BLACK;
    const Bitboard own = position.pieces(us);
    const Bitboard enemy = position.pieces(~us);
    const Bitboard enemyKing = position.pieces(~us, KING);
    const Bitboard occupied = position.pieces();

    TacticalMoveList list;

    Bitboard pieces = own & position.pieces(PAWN);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::PawnAttacks[us][from] & enemy;
        while (targets) {
            const Square to = pop_lsb(targets);
            if (rank_of(to) == (us == WHITE ? 7 : 0)) addTacticalPromotions(list, from, to, us);
            else addTacticalMove(list, Move{from, to});
        }
        if (position.ep_square() != SQ_NONE &&
            (Bitboards::PawnAttacks[us][from] & square_bb(position.ep_square()))) {
            Move move;
            move.from = from;
            move.to = position.ep_square();
            move.flags = Move::EN_PASSANT;
            addTacticalMove(list, move);
        }

        const int step = us == WHITE ? 8 : -8;
        const Square to = Square(int(from) + step);
        if (to < SQUARE_NB && position.piece_on(to) == EMPTY
            && rank_of(to) == (us == WHITE ? 7 : 0))
            addTacticalPromotions(list, from, to, us);
    }

    pieces = own & position.pieces(KNIGHT);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::KnightAttacks[from] & enemy & ~enemyKing;
        while (targets) addTacticalMove(list, Move{from, pop_lsb(targets)});
    }

    pieces = own & position.pieces(BISHOP);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::bishop_attacks(from, occupied) & enemy & ~enemyKing;
        while (targets) addTacticalMove(list, Move{from, pop_lsb(targets)});
    }

    pieces = own & position.pieces(ROOK);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::rook_attacks(from, occupied) & enemy & ~enemyKing;
        while (targets) addTacticalMove(list, Move{from, pop_lsb(targets)});
    }

    pieces = own & position.pieces(QUEEN);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::queen_attacks(from, occupied) & enemy & ~enemyKing;
        while (targets) addTacticalMove(list, Move{from, pop_lsb(targets)});
    }

    pieces = own & position.pieces(KING);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard targets = Bitboards::KingAttacks[from] & enemy & ~enemyKing;
        while (targets) addTacticalMove(list, Move{from, pop_lsb(targets)});
    }

    return list;
}

bool isCheckmate(Position& board, bool whiteToMove) {
    return board.isKingInCheck(whiteToMove) && generateLegalMoves(board, whiteToMove).empty();
}
