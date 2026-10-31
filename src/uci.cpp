#include "uci.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "movegen.h"
#include "piece.h"
#include "position.h"
#include "repetition.h"
#include "search.h"
#include "search_params.h"
#include "thread.h"

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

void emitRepetitionTrace(const Position& position, const char* event)
{
    static const bool enabled = std::getenv("ZERO_REP_TRACE") != nullptr;
    if (!enabled)
        return;
    const int count = Zero::Rules::repetitionCount(position);
    std::cerr
        << "[REP-TRACE] " << event
        << " root_count=" << count
        << " threefold=" << (Zero::Rules::isThreefoldRepetition(position) ? 1 : 0)
        << " halfmove=" << position.halfmove_clock()
        << '\n';
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

// Deterministic fixed-depth benchmark used to prove that a change keeps the search tree identical.
//   bench [depth] [file]    file: one FEN/EPD position per line (first four fields are used)
// Prints one line per position (best move, score, nodes) and a signature over all of them.
void runBench(const std::string& command, Zero::ThreadPool& threads, Position& position)
{
    std::istringstream args(command);
    std::string token;
    std::string path;
    int depth = 8;
    args >> token;
    if (args >> token) depth = std::max(1, std::atoi(token.c_str()));
    args >> path;

    std::vector<std::string> fens;
    if (!path.empty()) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            fens.push_back(line);
        }
    }
    if (fens.empty())
        fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -");

    auto& worker = threads.mainThread().worker();
    uint64_t totalNodes = 0;
    uint64_t signature = 1469598103934665603ULL;   // FNV-1a
    auto mix = [&](uint64_t v) { signature ^= v; signature *= 1099511628211ULL; };

    const auto t0 = std::chrono::steady_clock::now();
    int index = 0;
    for (const std::string& entry : fens) {
        std::istringstream ls(entry);
        std::string f0, f1, f2, f3;
        ls >> f0 >> f1 >> f2 >> f3;
        if (!position.loadFEN(f0 + " " + f1 + " " + f2 + " " + f3 + " 0 1"))
            continue;
        ++index;

        worker.clearHash();
        worker.clearHistory();
        worker.clearKillers();

        Zero::Search::Limits limits;
        limits.depth = depth;

        std::ostringstream sink;
        std::streambuf* previous = std::cout.rdbuf(sink.rdbuf());
        const auto result = worker.iterativeDeepening(limits);
        std::cout.rdbuf(previous);

        const uint64_t nodes = worker.nodes();
        totalNodes += nodes;
        const std::string best = toUciMove(result.bestMove);
        std::cout << "bench " << index << " " << best << " " << result.score << " " << nodes << "\n";
        for (char c : best) mix(static_cast<uint64_t>(c));
        mix(static_cast<uint64_t>(static_cast<int64_t>(result.score)));
        mix(nodes);
    }
    const auto ms = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count());
    std::cout << "bench total positions " << index << " nodes " << totalNodes << " time " << ms
              << " nps " << (totalNodes * 1000 / static_cast<uint64_t>(ms))
              << " signature " << std::hex << signature << std::dec << std::endl;
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
            std::cout << "id name ZERO 2.6 R3.5 QS-L0\n";
            std::cout << "id author Rohan Singh\n";
            std::cout << "option name Hash type spin default 16 min 1 max 4096\n";
            std::cout << "option name Move Overhead type spin default 20 min 0 max 2000\n";
            {
                const Zero::Search::SearchParams defaults;
                for (const auto& def : Zero::Search::searchParamTable())
                    std::cout << "option name " << def.name << " type spin default "
                              << defaults.*(def.field) << " min " << def.lo
                              << " max " << def.hi << "\n";
            }
            std::cout << "option name LMRLogScale type spin default 470 min 0 max 1200\n";
            std::cout << "option name LMRBase type spin default 820 min -4096 max 4096\n";
            std::cout << "option name LMRDepthCoeff type spin default 0 min -2048 max 2048\n";
            std::cout << "option name LMRMoveCoeff type spin default 0 min -2048 max 2048\n";
            std::cout << "option name LMRPVAdjust type spin default -512 min -2048 max 2048\n";
            std::cout << "option name LMRCutAdjust type spin default 512 min -2048 max 2048\n";
            std::cout << "option name LMRTTAdjust type spin default -1024 min -4096 max 1024\n";
            std::cout << "option name LMRHistoryCoeff type spin default -80 min -1024 max 1024\n";
            std::cout << "option name LMRContinuationCoeff type spin default 0 min -1024 max 1024\n";
            std::cout << "option name LMRImprovingAdjust type spin default -256 min -2048 max 1024\n";
            std::cout << "option name LMRTacticalSafetyAdjust type spin default 0 min -2048 max 1024\n";
            std::cout << "uciok\n" << std::flush;
        } else if (command == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (command == "ucinewgame") {
            position = Position();
            gameStates.clear();
            threads.mainThread().worker().clearHash();
            threads.mainThread().worker().clearHistory();
            threads.mainThread().worker().clearKillers();
        } else if (command.rfind("setoption name ", 0) == 0) {
            const std::string body = command.substr(std::string("setoption name ").size());
            const size_t valuePos = body.find(" value ");
            if (valuePos != std::string::npos) {
                const std::string name = body.substr(0, valuePos);
                try {
                    const int value = std::stoi(body.substr(valuePos + 7));
                    auto& worker = threads.mainThread().worker();
                    if (name == "Hash")                        worker.setHashSize(static_cast<size_t>(std::max(1, value)));
                    else if (name == "LMRBase")                worker.setLMRBase(value);
                    else if (name == "LMRDepthCoeff")          worker.setLMRDepthCoeff(value);
                    else if (name == "LMRMoveCoeff")           worker.setLMRMoveCoeff(value);
                    else if (name == "LMRPVAdjust")            worker.setLMRPVAdjust(value);
                    else if (name == "LMRCutAdjust")           worker.setLMRCutAdjust(value);
                    else if (name == "LMRTTAdjust")            worker.setLMRTTAdjust(value);
                    else if (name == "LMRHistoryCoeff")        worker.setLMRHistoryCoeff(value);
                    else if (name == "LMRContinuationCoeff")   worker.setLMRContinuationCoeff(value);
                    else if (name == "LMRImprovingAdjust")     worker.setLMRImprovingAdjust(value);
                    else if (name == "LMRTacticalSafetyAdjust") worker.setLMRTacticalSafetyAdjust(value);
                    else                                       worker.setParam(name, value);
                } catch (...) {
                    // Ignore malformed UCI option values.
                }
            }
        } else if (command.rfind("position ", 0) == 0) {
            setPosition(position, gameStates, command);
            emitRepetitionTrace(position, "after-position");
        } else if (command.rfind("bench", 0) == 0) {
            runBench(command, threads, position);
            position = Position();
            gameStates.clear();
        } else if (command.rfind("go", 0) == 0) {
            emitRepetitionTrace(position, "before-go");
            const auto limits = parseGoLimits(command, position.isWhiteToMove());
            auto result = threads.mainThread().worker().iterativeDeepening(limits);

            if (!result.bestMove.isOk()) {
                const auto legal = generateLegalMoves(position, position.isWhiteToMove());
                if (legal.empty()) {
                    // Terminal root: report it in a form GUIs and tuners can read.
                    const bool mated = position.isKingInCheck(position.isWhiteToMove());
                    std::cout << "info depth 0 score " << (mated ? "mate 0" : "cp 0") << '\n';
                } else {
                    // Search was cut off before finishing depth 1: never output an
                    // illegal "0000" when a legal move exists.
                    result.bestMove = legal.front();
                }
            }
            std::cout << "bestmove " << toUciMove(result.bestMove) << '\n' << std::flush;
        } else if (command == "quit") {
            break;
        }
    }
}
