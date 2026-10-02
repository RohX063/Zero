#include "uci.h"

#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>

#include "movegen.h"
#include "piece.h"
#include "position.h"
#include "thread.h"
#include "search.h"

namespace {

std::string toUciMove(const Move& move)
{
    if (!move.isOk()) return "0000";
    std::string result;
    result += char('a' + Zero::file_of(move.from));
    result += char('1' + Zero::rank_of(move.from));
    result += char('a' + Zero::file_of(move.to));
    result += char('1' + Zero::rank_of(move.to));
    if (move.isPromotion()) {
        switch (type_of(move.promotionPiece)) {
            case Zero::QUEEN: result += 'q'; break;
            case Zero::ROOK: result += 'r'; break;
            case Zero::BISHOP: result += 'b'; break;
            case Zero::KNIGHT: result += 'n'; break;
            default: break;
        }
    }
    return result;
}

bool applyMove(Position& position, std::deque<StateInfo>& states, const std::string& text)
{
    for (const Move& legal : generateLegalMoves(position, position.isWhiteToMove())) {
        if (toUciMove(legal) != text) continue;
        states.emplace_back();
        position.doMove(legal, states.back());
        return true;
    }
    return false;
}

void setPosition(Position& position, std::deque<StateInfo>& states, const std::string& command)
{
    constexpr const char* prefix = "position ";
    if (command.rfind(prefix, 0) != 0) return;
    std::string body = command.substr(9);

    if (body.rfind("startpos", 0) == 0) {
        position = Position();
        states.clear();
        body = body.substr(8);
    } else if (body.rfind("fen ", 0) == 0) {
        const std::string rest = body.substr(4);
        const size_t movesPos = rest.find(" moves ");
        const std::string fen = movesPos == std::string::npos ? rest : rest.substr(0, movesPos);
        if (!position.loadFEN(fen)) position = Position();
        states.clear();
        if (movesPos != std::string::npos) {
            std::stringstream ss(rest.substr(movesPos + 7));
            std::string moveText;
            while (ss >> moveText) if (!applyMove(position, states, moveText)) break;
        }
        return;
    } else {
        return;
    }

    if (!body.empty() && body.front() == ' ') body.erase(body.begin());
    const size_t movesPos = body.find("moves ");
    if (movesPos == std::string::npos) return;
    std::stringstream ss(body.substr(movesPos + 6));
    std::string moveText;
    while (ss >> moveText) if (!applyMove(position, states, moveText)) break;
}

Zero::Search::Limits parseGoLimits(const std::string& command, bool whiteToMove)
{
    Zero::Search::Limits limits;
    std::stringstream ss(command);
    std::string token;
    int whiteTime = 0;
    int blackTime = 0;
    int whiteIncrement = 0;
    int blackIncrement = 0;

    while (ss >> token) {
        if (token == "depth" && ss >> token)
            limits.depth = std::max(1, std::stoi(token));
        else if (token == "movetime" && ss >> token)
            limits.movetimeMs = std::max(1, std::stoi(token));
        else if (token == "wtime" && ss >> token)
            whiteTime = std::max(0, std::stoi(token));
        else if (token == "btime" && ss >> token)
            blackTime = std::max(0, std::stoi(token));
        else if (token == "winc" && ss >> token)
            whiteIncrement = std::max(0, std::stoi(token));
        else if (token == "binc" && ss >> token)
            blackIncrement = std::max(0, std::stoi(token));
        else if (token == "movestogo" && ss >> token)
            limits.movesToGo = std::max(0, std::stoi(token));
    }

    limits.sideTimeMs = whiteToMove ? whiteTime : blackTime;
    limits.incrementMs = whiteToMove ? whiteIncrement : blackIncrement;
    return limits;
}

}

void uciLoop()
{
    Position::init();
    Position position;
    std::deque<StateInfo> gameStates;
    Zero::ThreadPool threads(position);

    std::string command;
    while (std::getline(std::cin, command)) {
        if (command == "uci") {
            std::cout << "id name Zero-Phase1-Bitboard\n";
            std::cout << "id author Rohan Singh\n";
            std::cout << "option name Hash type spin default 16 min 1 max 4096\n";
            std::cout << "uciok\n";
        } else if (command == "isready") {
            std::cout << "readyok\n";
        } else if (command == "ucinewgame") {
            position = Position();
            gameStates.clear();
            threads.mainThread().worker().clearHash();
            threads.mainThread().worker().clearHistory();
            threads.mainThread().worker().clearKillers();
        } else if (command.rfind("setoption name Hash value ", 0) == 0) {
            try {
                const std::string value = command.substr(std::string("setoption name Hash value ").size());
                threads.mainThread().worker().setHashSize(std::stoul(value));
            } catch (...) {
                // Ignore malformed UCI option values.
            }
        } else if (command.rfind("position ", 0) == 0) {
            setPosition(position, gameStates, command);
        } else if (command.rfind("go", 0) == 0) {
            const auto limits = parseGoLimits(command, position.isWhiteToMove());
            const auto result = threads.mainThread().worker().iterativeDeepening(limits);
            std::cout << "bestmove " << toUciMove(result.bestMove) << '\n';
        } else if (command == "quit") {
            break;
        }
    }
}
