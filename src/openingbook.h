#pragma once

#include <string>
#include <vector>
#include "move.h"
#include "board.h"
#include "polyglot.h"

class OpeningBook
{
public:
    bool load(const std::string& filename);
    bool hasMove(const Board& board) const;
    Move getMove(const Board& board) const;

    size_t size() const
{
    return entries.size();
}

private:
    std::vector<PolyglotEntry> entries;
    bool loaded = false;
};