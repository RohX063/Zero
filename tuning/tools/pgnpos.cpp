// pgnpos: replay cutechess PGN games with ZERO's own move generator.
//   pgnpos file.pgn show GAME PLY      print FEN before/after the PLY-th ply (1-based) of round GAME
//   pgnpos file.pgn mine out.epd       list sacrifice candidates (material given up, engine still happy)
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "movegen.h"
#include "piece.h"
#include "position.h"
using namespace Zero;

struct Game { int round = 0; std::string white, black, fen, result; std::vector<std::string> san, com; };

static std::string fenOf(const Position& p, int fullmove)
{
    static const char sym[] = " PNBRQKpnbrqk";
    std::string f;
    for (int r = 7; r >= 0; --r) {
        int e = 0;
        for (int fl = 0; fl < 8; ++fl) {
            Piece pc = p.piece_on(make_square(fl, r));
            if (pc == EMPTY) { ++e; continue; }
            if (e) { f += char('0' + e); e = 0; }
            f += sym[pc];
        }
        if (e) f += char('0' + e);
        if (r) f += '/';
    }
    const StateInfo* s = p.state();
    f += p.isWhiteToMove() ? " w " : " b ";
    std::string c;
    if (s->whiteKingSideCastle) c += 'K';
    if (s->whiteQueenSideCastle) c += 'Q';
    if (s->blackKingSideCastle) c += 'k';
    if (s->blackQueenSideCastle) c += 'q';
    f += c.empty() ? "-" : c;
    f += ' ';
    if (s->epSquare == SQ_NONE) f += '-';
    else { f += char('a' + file_of(s->epSquare)); f += char('1' + rank_of(s->epSquare)); }
    f += ' ' + std::to_string(p.halfmove_clock()) + ' ' + std::to_string(fullmove);
    return f;
}

static std::string uci(const Move& m)
{
    std::string t;
    t += char('a' + file_of(m.from_sq())); t += char('1' + rank_of(m.from_sq()));
    t += char('a' + file_of(m.to_sq()));   t += char('1' + rank_of(m.to_sq()));
    return t;
}

static std::vector<Game> readPgn(const std::string& path)
{
    std::ifstream in(path); std::string line; std::vector<Game> games; Game* g = nullptr; std::string text;
    auto flush = [&]() {
        if (!g) return;
        size_t i = 0; std::string tok;
        while (i < text.size()) {
            char c = text[i];
            if (c == '{') { size_t j = text.find('}', i); std::string cm = text.substr(i + 1, j - i - 1);
                if (!g->san.empty() && g->com.size() < g->san.size()) { g->com.resize(g->san.size()); g->com[g->san.size() - 1] = cm; }
                i = j + 1; continue; }
            if (isspace((unsigned char)c)) { i++; continue; }
            size_t j = i; while (j < text.size() && !isspace((unsigned char)text[j]) && text[j] != '{') j++;
            tok = text.substr(i, j - i); i = j;
            if (tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*") continue;
            size_t dots = tok.find_last_of('.'); if (dots != std::string::npos) tok = tok.substr(dots + 1);
            while (!tok.empty() && (tok.back() == '+' || tok.back() == '#' || tok.back() == '!' || tok.back() == '?')) tok.pop_back();
            if (tok.empty()) continue;
            g->san.push_back(tok);
        }
        g->com.resize(g->san.size());
        text.clear();
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("[Event ", 0) == 0) { flush(); games.emplace_back(); g = &games.back(); }
        else if (line.rfind('[', 0) == 0 && g) {
            auto val = [&]() { size_t a = line.find('"'), b = line.rfind('"'); return line.substr(a + 1, b - a - 1); };
            if (line.rfind("[Round ", 0) == 0) g->round = std::atoi(val().c_str());
            else if (line.rfind("[White ", 0) == 0) g->white = val();
            else if (line.rfind("[Black ", 0) == 0) g->black = val();
            else if (line.rfind("[FEN ", 0) == 0) g->fen = val();
            else if (line.rfind("[Result ", 0) == 0) g->result = val();
        } else if (g) text += line + ' ';
    }
    flush();
    return games;
}

static bool pickMove(Position& pos, const std::string& san, Move& out)
{
    auto moves = generateLegalMoves(pos, pos.isWhiteToMove());
    if (san.rfind("O-O", 0) == 0) {
        bool longc = san.rfind("O-O-O", 0) == 0;
        for (const Move& m : moves)
            if (type_of(pos.piece_on(m.from_sq())) == KING && std::abs(file_of(m.to_sq()) - file_of(m.from_sq())) == 2
                && ((file_of(m.to_sq()) == 2) == longc)) { out = m; return true; }
        return false;
    }
    std::string s = san; PieceType pt = PAWN; size_t k = 0;
    if (isupper((unsigned char)s[0]) && s[0] != 'O') { switch (s[0]) { case 'N': pt = KNIGHT; break; case 'B': pt = BISHOP; break;
        case 'R': pt = ROOK; break; case 'Q': pt = QUEEN; break; case 'K': pt = KING; break; } k = 1; }
    PieceType promo = NO_PIECE_TYPE;
    size_t eq = s.find('='); if (eq != std::string::npos) { switch (s[eq + 1]) { case 'N': promo = KNIGHT; break; case 'B': promo = BISHOP; break;
        case 'R': promo = ROOK; break; case 'Q': promo = QUEEN; break; } s = s.substr(0, eq); }
    s.erase(std::remove(s.begin(), s.end(), 'x'), s.end());
    if (s.size() < k + 2) return false;
    int tf = s[s.size() - 2] - 'a', tr = s[s.size() - 1] - '1';
    std::string dis = s.substr(k, s.size() - 2 - k);
    for (const Move& m : moves) {
        if (type_of(pos.piece_on(m.from_sq())) != pt) continue;
        if (file_of(m.to_sq()) != tf || rank_of(m.to_sq()) != tr) continue;
        if ((promo != NO_PIECE_TYPE) != m.isPromotion()) continue;
        if (promo != NO_PIECE_TYPE && type_of(m.promotionPiece) != promo) continue;
        bool ok = true;
        for (char d : dis) { if (d >= 'a' && d <= 'h' && file_of(m.from_sq()) != d - 'a') ok = false;
                             if (d >= '1' && d <= '8' && rank_of(m.from_sq()) != d - '1') ok = false; }
        if (ok) { out = m; return true; }
    }
    return false;
}

static int material(const Position& p, Color c)
{
    return popcount(p.pieces(c, PAWN)) + 3 * popcount(p.pieces(c, KNIGHT)) + 3 * popcount(p.pieces(c, BISHOP))
         + 5 * popcount(p.pieces(c, ROOK)) + 9 * popcount(p.pieces(c, QUEEN));
}

static double evalOf(const std::string& c)  // "+0.31/7 0.55s" -> 0.31, depth via ref
{ return c.empty() ? 0.0 : std::atof(c.c_str()); }


static int pv(Piece pc) { switch (type_of(pc)) { case PAWN: return 1; case KNIGHT: case BISHOP: return 3; case ROOK: return 5; case QUEEN: return 9; default: return 0; } }

// Best net material the side to move can win by one capture followed by at most one recapture.
// Returns 0 when nothing is en prise. (Crude one-step exchange estimate, not full SEE.)
static int hangingNet(Position& pos, int* victimSq = nullptr)
{
    int best = 0;
    auto moves = generateLegalMoves(pos, pos.isWhiteToMove());
    for (const Move& m : moves) {
        Piece victim = pos.piece_on(m.to_sq());
        if (victim == EMPTY) continue;
        int v = pv(victim); if (v < 3) continue;           // only pieces and better count as offers
        int a = pv(pos.piece_on(m.from_sq()));
        StateInfo st; pos.doMove(m, st);
        bool recapture = false;
        for (const Move& r : generateLegalMoves(pos, pos.isWhiteToMove())) if (r.to_sq() == m.to_sq()) { recapture = true; break; }
        pos.undoMove(m);
        int net = recapture ? v - a : v;
        if (net > best) { best = net; if (victimSq) *victimSq = m.to_sq(); }
    }
    return best;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::cerr << "usage\n"; return 1; }
    Position::init();
    auto games = readPgn(argv[1]);
    std::string mode = argv[2];
    if (mode == "show") {
        int gi = std::atoi(argv[3]), ply = std::atoi(argv[4]);
        for (auto& g : games) if (g.round == gi) {
            Position pos; pos.loadFEN(g.fen); std::deque<StateInfo> st;
            int full0 = 1;
            for (int i = 0; i < (int)g.san.size() && i < ply; ++i) {
                Move m;
                if (!pickMove(pos, g.san[i], m)) { std::cerr << "cannot parse " << g.san[i] << " at ply " << i + 1 << "\n"; return 2; }
                if (i == ply - 1) {
                    int fullBefore = full0 + i / 2;
                    std::cout << "game " << gi << "  " << g.white << " (W) vs " << g.black << " (B)  result " << g.result << "\n";
                    std::cout << "ply " << ply << ": " << g.san[i] << "  (" << uci(m) << ")  comment: " << g.com[i] << "\n";
                    std::cout << "FEN before: " << fenOf(pos, fullBefore) << "\n";
                }
                st.emplace_back(); pos.doMove(m, st.back());
            }
        }
        return 0;
    }

    if (mode == "offers") {
        std::ofstream out2(argv[3]); int found = 0, ignored = 0, declined = 0;
        for (auto& g : games) {
            Position pos; if (!pos.loadFEN(g.fen)) continue; std::deque<StateInfo> st;
            std::vector<Move> mv; std::vector<std::string> fens; std::vector<int> net, before, sqv;
            for (int i = 0; i < (int)g.san.size(); ++i) {
                Move m; if (!pickMove(pos, g.san[i], m)) break;
                fens.push_back(fenOf(pos, 1 + i / 2)); mv.push_back(m);
                // threat already present before the move? (opponent to move after a null move)
                StateInfo ns; pos.doNullMove(ns); int b = hangingNet(pos); pos.undoNullMove();
                st.emplace_back(); pos.doMove(m, st.back());
                int sq = -1; int n = hangingNet(pos, &sq);
                before.push_back(b); net.push_back(n); sqv.push_back(sq);
            }
            int n = (int)mv.size();
            for (int i = 0; i + 1 < n; ++i) {
                if (net[i] < 2) continue;
                double ev = evalOf(g.com[i]);
                if (ev < -0.30) continue;                        // engine itself must not think it is losing
                bool tookIt = (mv[i + 1].to_sq() == sqv[i]);     // opponent captured on the offered square
                bool wasThreat = before[i] >= 2;                 // piece was already attacked before the move
                if (tookIt) continue;                            // realised sacrifices are mined elsewhere
                out2 << fens[i].substr(0, fens[i].rfind(' ', fens[i].rfind(' ') - 1)) << " sac " << g.san[i] << "; id \"r" << g.round << "_p" << i + 1
                     << "\"; uci " << uci(mv[i]) << "; c0 \"declined offer net " << net[i] << (wasThreat ? " ignored-threat" : " placed-en-prise")
                     << " eval " << g.com[i] << " " << (i % 2 == 0 ? g.white : g.black) << "\";\n";
                ++found; if (wasThreat) ++ignored; ++declined;
            }
        }
        std::cerr << "declined offers " << found << " (ignored threat " << ignored << ", placed en prise " << found - ignored << ")\n";
        return 0;
    }
    // mine
    std::ofstream out(argv[3]);
    int found = 0;
    for (auto& g : games) {
        Position pos; if (!pos.loadFEN(g.fen)) continue; std::deque<StateInfo> st;
        std::vector<std::string> fens; std::vector<int> matW, matB; std::vector<Move> mv;
        for (int i = 0; i < (int)g.san.size(); ++i) {
            Move m; if (!pickMove(pos, g.san[i], m)) break;
            fens.push_back(fenOf(pos, 1 + i / 2)); matW.push_back(material(pos, WHITE)); matB.push_back(material(pos, BLACK)); mv.push_back(m);
            st.emplace_back(); pos.doMove(m, st.back());
        }
        int n = (int)mv.size();
        for (int i = 0; i + 4 < n; ++i) {
            bool whiteMoved = (i % 2 == 0);
            int before = whiteMoved ? matW[i] - matB[i] : matB[i] - matW[i];
            int after2 = whiteMoved ? matW[i + 2] - matB[i + 2] : matB[i + 2] - matW[i + 2];   // after opponent's reply
            int after4 = whiteMoved ? matW[i + 4] - matB[i + 4] : matB[i + 4] - matW[i + 4];
            int deficit = std::max(after2, after4) - before;   // still down after two more plies
            double ev = evalOf(g.com[i]);
            // engine itself must have been happy (mover view), and the sacrifice must pay back later or keep the eval
            if (deficit <= -2 && ev >= 0.30 && !(g.san[i].find('x') != std::string::npos && deficit == -2 && after4 - before > -2)) {
                int later = whiteMoved ? matW[std::min(n - 1, i + 8)] - matB[std::min(n - 1, i + 8)] : matB[std::min(n - 1, i + 8)] - matW[std::min(n - 1, i + 8)];
                bool nowWhite = true; (void)nowWhite;
                out << fens[i].substr(0, fens[i].rfind(' ', fens[i].rfind(' ') - 1)) << " sac " << g.san[i] << "; id \"r" << g.round << "_p" << i + 1 << "\"; uci " << uci(mv[i]) << "; c0 \"deficit " << deficit
                    << " eval " << g.com[i] << " recoup8 " << (later - before) << " " << (whiteMoved ? g.white : g.black) << "\";\n";
                ++found;
            }
        }
    }
    std::cerr << "games " << games.size() << " candidates " << found << "\n";
    return 0;
}
