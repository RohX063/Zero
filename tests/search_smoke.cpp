#include "position.h"
#include "search.h"
#include <iostream>

int main() {
    Position::init();
    Position position;
    Zero::Search::Worker worker(position);
    const auto result = worker.searchRoot(5);
    std::cout << "bestmove ";
    if (result.bestMove.isOk()) {
        std::cout << char('a' + Zero::file_of(result.bestMove.from))
                  << char('1' + Zero::rank_of(result.bestMove.from))
                  << char('a' + Zero::file_of(result.bestMove.to))
                  << char('1' + Zero::rank_of(result.bestMove.to));
    } else std::cout << "0000";
    std::cout << " score " << result.score << " nodes " << worker.nodes() << '\n';
    return result.bestMove.isOk() ? 0 : 1;
}
