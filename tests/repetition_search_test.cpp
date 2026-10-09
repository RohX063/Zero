#include <cassert>
#include <deque>
#include <string>

#include "move.h"
#include "movegen.h"
#include "position.h"
#include "repetition.h"
#include "search.h"

namespace {

Move findMove(Position& pos, const std::string& uci)
{
    for (const auto& move : generateLegalMoves(pos, pos.isWhiteToMove())) {
        std::string text;
        text += char('a' + Zero::file_of(move.from));
        text += char('1' + Zero::rank_of(move.from));
        text += char('a' + Zero::file_of(move.to));
        text += char('1' + Zero::rank_of(move.to));
        if (move.isPromotion()) text += 'q';
        if (text == uci) return move;
    }
    return Move::none();
}

} // namespace

int main()
{
    Position::init();

    Position position;
    assert(position.loadFEN("1q2r1k1/6p1/1P6/7Q/6P1/4p2P/1PP2P2/3R2K1 w - - 0 67"));

    std::deque<StateInfo> history;
    const char* cycle[] = {"h5d5", "g8h7", "d5h5", "h7g8", "h5d5", "g8h7", "d5h5"};
    auto play = [&](const char* text) {
        const auto move = findMove(position, text);
        assert(move.isOk());
        history.emplace_back();
        position.doMove(move, history.back());
    };

    // After six plies the position equals the one after ply two (count 2).
    for (int i = 0; i < 6; ++i)
        play(cycle[i]);
    assert(Zero::Rules::repetitionCount(position) == 2);

    // Playing d5h5 reaches the position after ply three: only its SECOND
    // occurrence, so it must not be reported as a threefold.
    const auto notYet = findMove(position, "d5h5");
    assert(notYet.isOk());
    assert(!Zero::Rules::moveCausesThreefold(position, notYet));

    // Search-tree rule: treated as if the game started at the initial position
    // (ply 6 below the root) the repeat lies inside the tree -> draw; with the
    // current position as root (ply 0) it only repeats history once -> no draw.
    assert(Zero::Rules::isDrawInSearch(position, 6));
    assert(!Zero::Rules::isDrawInSearch(position, 0));

    // One more cycle move: black to move, and h7g8 would be the third occurrence.
    play(cycle[6]);
    assert(Zero::Rules::repetitionCount(position) == 2);
    const auto repeating = findMove(position, "h7g8");
    assert(repeating.isOk());
    assert(Zero::Rules::moveCausesThreefold(position, repeating));

    for (const auto& legal : generateLegalMoves(position, position.isWhiteToMove())) {
        if (legal.from == repeating.from && legal.to == repeating.to)
            continue;
        assert(!Zero::Rules::moveCausesThreefold(position, legal));
        break;
    }

    // Insufficient material and the fifty-move rule.
    Position bare;
    assert(bare.loadFEN("8/8/8/4k3/8/8/4N3/4K3 w - - 0 1"));
    assert(Zero::Rules::isInsufficientMaterial(bare));
    assert(Zero::Rules::isDrawInSearch(bare, 3));
    Position rook;
    assert(rook.loadFEN("8/8/8/4k3/8/8/4R3/4K3 w - - 0 1"));
    assert(!Zero::Rules::isInsufficientMaterial(rook));
    Position fifty;
    assert(fifty.loadFEN("8/8/8/4k3/8/8/4R3/4K3 w - - 100 80"));
    assert(Zero::Rules::isDrawInSearch(fifty, 3));

    // The search must still complete on a position with repetition history.
    Zero::Search::Worker worker(position);
    const auto result = worker.searchRoot(4);
    assert(result.completed);
    assert(result.bestMove.isOk());
    return 0;
}
