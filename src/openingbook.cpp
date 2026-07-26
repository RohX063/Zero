#include "openingbook.h"

#include <fstream>

bool OpeningBook::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        return false;

    entries.clear();

    PolyglotEntry entry;

    while(file.read(reinterpret_cast<char*>(&entry), sizeof(entry)))
    {
        entries.push_back(entry);
    }

    loaded = true;

    return true;
}
bool OpeningBook::hasMove(const Board& board) const
{
    uint64_t key = computePolyglotKey(board);

    for (const auto& entry : entries)
    {
        if (entry.key == key)
            return true;
    }

    return false;
}
Move OpeningBook::getMove(const Board& board) const
{
    uint64_t key = computePolyglotKey(board);

    for (const auto& entry : entries)
    {
        if (entry.key == key)
        {
            // TODO:
            // Convert Polyglot move -> Zero Move
            break;
        }
    }

    return Move();
}