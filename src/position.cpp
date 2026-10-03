#include "position.h"
#include "piece.h"
#include "zobrist.h"
#include <cctype>
#include <iostream>
#include <sstream>

using namespace Zero;

Position::Position() { initialize(); }

Position::Position(const Position& other) { *this = other; }

Position& Position::operator=(const Position& other) {
    if (this == &other) return *this;
    board_ = other.board_;
    byPiece_ = other.byPiece_;
    byColor_ = other.byColor_;
    byType_ = other.byType_;
    occupancy_ = other.occupancy_;
    rootState_ = *other.state_;
    rootState_.previous = nullptr;
    rootState_.isNullMove = false;
    state_ = &rootState_;
    return *this;
}

void Position::init() { Bitboards::init(); Zobrist::init(); }

bool Position::equals(const Position& other) const {
    return board_ == other.board_
        && byPiece_ == other.byPiece_
        && byColor_ == other.byColor_
        && byType_ == other.byType_
        && occupancy_ == other.occupancy_
        && state_->whiteToMove == other.state_->whiteToMove
        && state_->whiteKingSideCastle == other.state_->whiteKingSideCastle
        && state_->whiteQueenSideCastle == other.state_->whiteQueenSideCastle
        && state_->blackKingSideCastle == other.state_->blackKingSideCastle
        && state_->blackQueenSideCastle == other.state_->blackQueenSideCastle
        && state_->epSquare == other.state_->epSquare
        && state_->halfmoveClock == other.state_->halfmoveClock;
}

void Position::clearBoard() {
    board_.fill(EMPTY);
    byPiece_.fill(0);
    byColor_.fill(0);
    byType_.fill(0);
    occupancy_ = 0;
    state_->key = 0;
    state_->halfmoveClock = 0;
    state_->isNullMove = false;
}

void Position::putPiece(Square sq, Piece piece) {
    if (piece == EMPTY || sq >= SQUARE_NB) return;
    const Bitboard b = square_bb(sq);
    board_[sq] = piece;
    byPiece_[piece] |= b;
    state_->key ^= Zobrist::piece(piece, sq);
    byColor_[color_of(piece)] |= b;
    byType_[type_of(piece)] |= b;
    occupancy_ |= b;
}

void Position::removePiece(Square sq, Piece piece) {
    if (piece == EMPTY || sq >= SQUARE_NB) return;
    const Bitboard b = square_bb(sq);
    board_[sq] = EMPTY;
    byPiece_[piece] &= ~b;
    state_->key ^= Zobrist::piece(piece, sq);
    byColor_[color_of(piece)] &= ~b;
    byType_[type_of(piece)] &= ~b;
    occupancy_ &= ~b;
}

void Position::movePiece(Square from, Square to, Piece piece) {
    if (piece == EMPTY || from >= SQUARE_NB || to >= SQUARE_NB || from == to) return;

    const Bitboard delta = square_bb(from) | square_bb(to);
    state_->key ^= Zobrist::piece(piece, from) ^ Zobrist::piece(piece, to);
    board_[from] = EMPTY;
    board_[to] = piece;
    byPiece_[piece] ^= delta;
    byColor_[color_of(piece)] ^= delta;
    byType_[type_of(piece)] ^= delta;
    occupancy_ ^= delta;
}

void Position::setPiece(int row, int col, Piece piece) {
    setPiece(make_square(col, 7 - row), piece);
}

void Position::setPiece(Square sq, Piece piece) {
    if (sq >= SQUARE_NB) return;
    if (board_[sq] != EMPTY) removePiece(sq, board_[sq]);
    if (piece != EMPTY) putPiece(sq, piece);
}

bool Position::loadFEN(const std::string& fen) {
    rootState_ = StateInfo{};
    state_ = &rootState_;
    clearBoard();

    std::stringstream ss(fen);
    std::string boardPart, sidePart, castlePart, epPart;
    int halfmoveClock = 0, fullmoveNumber = 1;
    ss >> boardPart >> sidePart >> castlePart >> epPart >> halfmoveClock >> fullmoveNumber;
    if (boardPart.empty() || sidePart.empty() || castlePart.empty() || epPart.empty()) return false;

    int row = 0, col = 0;
    for (char ch : boardPart) {
        if (ch == '/') { ++row; col = 0; continue; }
        if (row >= 8 || col > 8) return false;
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            col += ch - '0';
            if (col > 8) return false;
            continue;
        }
        Piece piece = EMPTY;
        switch (ch) {
            case 'P': piece = WHITE_PAWN; break;
            case 'N': piece = WHITE_KNIGHT; break;
            case 'B': piece = WHITE_BISHOP; break;
            case 'R': piece = WHITE_ROOK; break;
            case 'Q': piece = WHITE_QUEEN; break;
            case 'K': piece = WHITE_KING; break;
            case 'p': piece = BLACK_PAWN; break;
            case 'n': piece = BLACK_KNIGHT; break;
            case 'b': piece = BLACK_BISHOP; break;
            case 'r': piece = BLACK_ROOK; break;
            case 'q': piece = BLACK_QUEEN; break;
            case 'k': piece = BLACK_KING; break;
            default: return false;
        }
        if (col >= 8) return false;
        setPiece(row, col, piece);
        ++col;
    }
    if (row != 7 || col != 8) return false;

    if (sidePart == "w") state_->whiteToMove = true;
    else if (sidePart == "b") state_->whiteToMove = false;
    else return false;

    if (castlePart != "-") {
        for (char c : castlePart) {
            switch (c) {
                case 'K': state_->whiteKingSideCastle = true; break;
                case 'Q': state_->whiteQueenSideCastle = true; break;
                case 'k': state_->blackKingSideCastle = true; break;
                case 'q': state_->blackQueenSideCastle = true; break;
                default: return false;
            }
        }
    }

    if (epPart != "-") {
        if (epPart.size() != 2 || epPart[0] < 'a' || epPart[0] > 'h' || epPart[1] < '1' || epPart[1] > '8')
            return false;
        state_->epSquare = make_square(epPart[0] - 'a', epPart[1] - '1');

        // Only retain an en-passant right when the side to move actually has
        // a pawn that can capture onto the EP square. This prevents irrelevant
        // FEN EP fields from creating false repetition-key differences.
        const Color capturer = state_->whiteToMove ? WHITE : BLACK;
        const Bitboard capturers =
            pieces(capturer, PAWN) & Bitboards::PawnAttacks[capturer][state_->epSquare];
        if (capturers == 0)
            state_->epSquare = SQ_NONE;
    }

    state_->halfmoveClock = std::max(0, halfmoveClock);
    state_->isNullMove = false;

    if (!state_->whiteToMove)
        state_->key ^= Zobrist::side;
    state_->key ^= Zobrist::castling_key(castlingRightsMask());
    state_->key ^= Zobrist::en_passant_key(state_->epSquare);
    return true;
}

void Position::initialize() {
    rootState_ = StateInfo{};
    state_ = &rootState_;
    clearBoard();

    constexpr Piece backRank[] = {
        WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN,
        WHITE_KING, WHITE_BISHOP, WHITE_KNIGHT, WHITE_ROOK
    };
    constexpr Piece blackBackRank[] = {
        BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN,
        BLACK_KING, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK
    };
    for (int f = 0; f < 8; ++f) {
        setPiece(make_square(f, 0), backRank[f]);
        setPiece(make_square(f, 1), WHITE_PAWN);
        setPiece(make_square(f, 6), BLACK_PAWN);
        setPiece(make_square(f, 7), blackBackRank[f]);
    }

    state_->whiteKingSideCastle = true;
    state_->whiteQueenSideCastle = true;
    state_->blackKingSideCastle = true;
    state_->blackQueenSideCastle = true;
    state_->halfmoveClock = 0;
    state_->isNullMove = false;
    state_->key ^= Zobrist::castling_key(castlingRightsMask());
}

void Position::printBoard() const {
    std::cout << "\n   +-----------------+\n";
    for (int row = 0; row < 8; ++row) {
        std::cout << 8 - row << "  |";
        for (int col = 0; col < 8; ++col) {
            const Piece p = getPiece(row, col);
            char c = '.';
            switch (p) {
                case WHITE_PAWN: c='P'; break; case WHITE_KNIGHT: c='N'; break;
                case WHITE_BISHOP: c='B'; break; case WHITE_ROOK: c='R'; break;
                case WHITE_QUEEN: c='Q'; break; case WHITE_KING: c='K'; break;
                case BLACK_PAWN: c='p'; break; case BLACK_KNIGHT: c='n'; break;
                case BLACK_BISHOP: c='b'; break; case BLACK_ROOK: c='r'; break;
                case BLACK_QUEEN: c='q'; break; case BLACK_KING: c='k'; break;
                default: break;
            }
            std::cout << ' ' << c;
        }
        std::cout << " |\n";
    }
    std::cout << "   +-----------------+\n      a b c d e f g h\n";
}

Square Position::king_square(Color c) const {
    Bitboard kings = byColor_[c] & byType_[KING];
    return kings ? lsb(kings) : SQ_NONE;
}

Bitboard Position::attackers_to(Square sq, Color by) const {
    const Bitboard occupied = occupancy_;
    Bitboard attackers = Bitboards::PawnAttacks[~by][sq] & byColor_[by] & byType_[PAWN];
    attackers |= Bitboards::KnightAttacks[sq] & byColor_[by] & byType_[KNIGHT];
    attackers |= Bitboards::KingAttacks[sq] & byColor_[by] & byType_[KING];
    attackers |= Bitboards::bishop_attacks(sq, occupied) & byColor_[by] & (byType_[BISHOP] | byType_[QUEEN]);
    attackers |= Bitboards::rook_attacks(sq, occupied) & byColor_[by] & (byType_[ROOK] | byType_[QUEEN]);
    return attackers;
}

bool Position::isSquareAttacked(Square sq, Color by) const {
    return attackers_to(sq, by) != 0;
}

bool Position::isSquareAttacked(int row, int col, bool byWhite) const {
    return isSquareAttacked(make_square(col, 7 - row), byWhite ? WHITE : BLACK);
}

bool Position::isKingInCheck(bool whiteKing) const {
    const Color c = whiteKing ? WHITE : BLACK;
    const Square king = king_square(c);
    return king != SQ_NONE && isSquareAttacked(king, ~c);
}

void Position::doMove(const Move& move, StateInfo& newState) {
    newState = *state_;
    newState.previous = state_;
    newState.isNullMove = false;
    newState.movedPiece = board_[move.from];
    newState.capturedPiece = board_[move.to];
    state_ = &newState;

    const int oldCastlingRights = castlingRightsMask();
    state_->key ^= Zobrist::en_passant_key(state_->epSquare);
    state_->epSquare = SQ_NONE;

    const Piece moving = newState.movedPiece;
    Piece captured = newState.capturedPiece;

    removePiece(move.from, moving);

    if (move.isEnPassant()) {
        const Square capSq = moving == WHITE_PAWN ? Square(move.to - 8) : Square(move.to + 8);
        captured = board_[capSq];
        if (captured != EMPTY) removePiece(capSq, captured);
    } else if (captured != EMPTY) {
        removePiece(move.to, captured);
    }

    const Piece placed = move.isPromotion() ? move.promotionPiece : moving;
    putPiece(move.to, placed);
    state_->capturedPiece = captured;

    if (moving == WHITE_PAWN && rank_of(move.from) == 1 && rank_of(move.to) == 3) {
        const Square ep = make_square(file_of(move.from), 2);
        const Bitboard capturers = pieces(BLACK, PAWN) & Bitboards::PawnAttacks[BLACK][ep];
        if (capturers)
            state_->epSquare = ep;
    } else if (moving == BLACK_PAWN && rank_of(move.from) == 6 && rank_of(move.to) == 4) {
        const Square ep = make_square(file_of(move.from), 5);
        const Bitboard capturers = pieces(WHITE, PAWN) & Bitboards::PawnAttacks[WHITE][ep];
        if (capturers)
            state_->epSquare = ep;
    }

    if (move.isCastle()) {
        const Color us = color_of(moving);
        const int rank = us == WHITE ? 0 : 7;
        const bool kingSide = move.isKingSideCastle();
        const Square rookFrom = make_square(kingSide ? 7 : 0, rank);
        const Square rookTo = make_square(kingSide ? 5 : 3, rank);
        const Piece rook = us == WHITE ? WHITE_ROOK : BLACK_ROOK;
        movePiece(rookFrom, rookTo, rook);
    }

    if (moving == WHITE_KING) {
        state_->whiteKingSideCastle = state_->whiteQueenSideCastle = false;
    } else if (moving == BLACK_KING) {
        state_->blackKingSideCastle = state_->blackQueenSideCastle = false;
    }
    setCastlingRightsForRookSquare(move.from);
    clearCastlingRightsForCapturedRook(move.to, captured);

    state_->halfmoveClock =
        (type_of(moving) == PAWN || captured != EMPTY)
            ? 0
            : state_->halfmoveClock + 1;

    state_->key ^= Zobrist::castling_key(oldCastlingRights);
    state_->key ^= Zobrist::castling_key(castlingRightsMask());
    state_->key ^= Zobrist::en_passant_key(state_->epSquare);
    state_->key ^= Zobrist::side;
    state_->whiteToMove = !state_->whiteToMove;
}

void Position::undoMove(const Move& move) {
    if (state_ == &rootState_ || state_->previous == nullptr) return;
    const StateInfo* current = state_;
    const Piece moving = current->movedPiece;
    const Piece placed = move.isPromotion() ? move.promotionPiece : moving;

    removePiece(move.to, placed);
    putPiece(move.from, moving);

    if (move.isEnPassant()) {
        const Square capSq = moving == WHITE_PAWN ? Square(move.to - 8) : Square(move.to + 8);
        if (current->capturedPiece != EMPTY) putPiece(capSq, current->capturedPiece);
    } else if (current->capturedPiece != EMPTY) {
        putPiece(move.to, current->capturedPiece);
    }

    if (move.isCastle()) {
        const Color us = color_of(moving);
        const int rank = us == WHITE ? 0 : 7;
        const bool kingSide = move.isKingSideCastle();
        const Square rookFrom = make_square(kingSide ? 7 : 0, rank);
        const Square rookTo = make_square(kingSide ? 5 : 3, rank);
        const Piece rook = us == WHITE ? WHITE_ROOK : BLACK_ROOK;
        movePiece(rookTo, rookFrom, rook);
    }

    state_ = current->previous;
}

void Position::doNullMove(StateInfo& newState) {
    newState = *state_;
    newState.previous = state_;
    newState.capturedPiece = EMPTY;
    newState.movedPiece = EMPTY;
    newState.isNullMove = true;
    state_ = &newState;
    state_->key ^= Zobrist::en_passant_key(state_->epSquare);
    state_->epSquare = SQ_NONE;
    state_->key ^= Zobrist::side;
    state_->whiteToMove = !state_->whiteToMove;
}

void Position::undoNullMove() {
    if (state_ == &rootState_ || state_->previous == nullptr) return;
    state_ = state_->previous;
}

void Position::setCastlingRightsForRookSquare(Square sq) {
    if (sq == make_square(0,0)) state_->whiteQueenSideCastle = false;
    if (sq == make_square(7,0)) state_->whiteKingSideCastle = false;
    if (sq == make_square(0,7)) state_->blackQueenSideCastle = false;
    if (sq == make_square(7,7)) state_->blackKingSideCastle = false;
}

void Position::clearCastlingRightsForCapturedRook(Square sq, Piece captured) {
    if (captured == WHITE_ROOK) {
        if (sq == make_square(0,0)) state_->whiteQueenSideCastle = false;
        if (sq == make_square(7,0)) state_->whiteKingSideCastle = false;
    } else if (captured == BLACK_ROOK) {
        if (sq == make_square(0,7)) state_->blackQueenSideCastle = false;
        if (sq == make_square(7,7)) state_->blackKingSideCastle = false;
    }
}

bool Position::canWhiteCastleKingSide() const { return state_->whiteKingSideCastle; }
bool Position::canWhiteCastleQueenSide() const { return state_->whiteQueenSideCastle; }
bool Position::canBlackCastleKingSide() const { return state_->blackKingSideCastle; }
bool Position::canBlackCastleQueenSide() const { return state_->blackQueenSideCastle; }

int Position::getEnPassantRow() const {
    return state_->epSquare == SQ_NONE ? -1 : 7 - rank_of(state_->epSquare);
}
int Position::getEnPassantCol() const { return state_->epSquare == SQ_NONE ? -1 : file_of(state_->epSquare); }

Piece Position::getPiece(int row, int col) const {
    return row >= 0 && row < 8 && col >= 0 && col < 8 ? board_[make_square(col, 7 - row)] : EMPTY;
}

bool Position::isWhiteToMove() const { return state_->whiteToMove; }
void Position::setSideToMove(bool white) {
    if (state_->whiteToMove != white)
        state_->key ^= Zobrist::side;
    state_->whiteToMove = white;
}
void Position::switchSide() {
    state_->key ^= Zobrist::side;
    state_->whiteToMove = !state_->whiteToMove;
}

int Position::castlingRightsMask() const {
    return (state_->whiteKingSideCastle ? 1 : 0)
         | (state_->whiteQueenSideCastle ? 2 : 0)
         | (state_->blackKingSideCastle ? 4 : 0)
         | (state_->blackQueenSideCastle ? 8 : 0);
}

Zero::Key Position::recomputeKey() const {
    Zero::Key result = 0;
    for (Square sq = 0; sq < SQUARE_NB; ++sq) {
        const Piece piece = board_[sq];
        if (piece != EMPTY)
            result ^= Zobrist::piece(piece, sq);
    }
    if (!state_->whiteToMove)
        result ^= Zobrist::side;
    result ^= Zobrist::castling_key(castlingRightsMask());
    result ^= Zobrist::en_passant_key(state_->epSquare);
    return result;
}
