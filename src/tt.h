#ifndef ZERO_TT_H
#define ZERO_TT_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "move.h"
#include "types.h"

namespace Zero::Search {

enum Bound : std::uint8_t {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = 3
};

struct TTData {
    Move move{};
    Value value = VALUE_DRAW;
    Depth depth = 0;
    Bound bound = BOUND_NONE;
    bool pv = false;
    bool hit = false;
};

class TranspositionTable {
public:
    static constexpr std::size_t DEFAULT_HASH_MB = 16;
    static constexpr std::size_t CLUSTER_SIZE = 4;

    explicit TranspositionTable(std::size_t megabytes = DEFAULT_HASH_MB);

    void resize(std::size_t megabytes);
    void clear();
    void new_search();

    TTData probe(Key key) const;
    void store(Key key, Value value, Bound bound, Depth depth, Move move, bool pv = false);

    std::size_t megabytes() const { return megabytes_; }
    std::uint32_t hashfull() const;

    static Value value_to_tt(Value value, int ply);
    static Value value_from_tt(Value value, int ply);

private:
    struct alignas(64) Cluster {
        struct Entry {
            std::uint32_t keyLo;
            std::uint32_t keyHi;
            std::int32_t value;
            std::uint16_t move;
            std::uint8_t depth;
            std::uint8_t meta;

            bool occupied() const { return keyLo != 0 || keyHi != 0; }
            bool matches(Key key) const { return keyLo == std::uint32_t(key) && keyHi == std::uint32_t(key >> 32); }
            Depth search_depth() const { return Depth(depth); }
            std::uint8_t generation() const { return meta & 0x1Fu; }
            Bound bound() const { return Bound((meta >> 5) & 0x3u); }
            bool pv() const { return (meta & 0x80u) != 0; }
        } entry[CLUSTER_SIZE];

        static_assert(sizeof(Entry) == 16, "TT entry must be exactly 16 bytes");
        static_assert(sizeof(Entry) * CLUSTER_SIZE == 64, "TT cluster payload must fit one cache line");
    };

    static constexpr std::uint8_t GENERATION_MASK = 31;

    static std::uint16_t pack_move(Move move);
    static Move unpack_move(std::uint16_t raw);
    static std::uint8_t pack_meta(std::uint8_t generation, Bound bound, bool pv);
    static std::uint8_t relative_age(std::uint8_t current, std::uint8_t stored);

    std::size_t cluster_index(Key key) const { return std::size_t(key) & cluster_mask_; }
    Cluster& cluster(Key key) { return table_[cluster_index(key)]; }
    const Cluster& cluster(Key key) const { return table_[cluster_index(key)]; }
    Cluster::Entry* replacement_slot(Cluster& cluster, Key key) const;

    std::unique_ptr<Cluster[]> table_;
    std::size_t cluster_count_ = 0;
    std::size_t cluster_mask_ = 0;
    std::size_t megabytes_ = 0;
    std::uint8_t generation_ = 0;
};

} // namespace Zero::Search

#endif
