#include "tt.h"

#include <algorithm>
#include <limits>

namespace Zero::Search {

namespace {
constexpr std::uint8_t PV_MASK = 0x80u;
constexpr std::size_t BYTES_PER_MB = 1024u * 1024u;
}

TranspositionTable::TranspositionTable(std::size_t megabytes) {
    resize(megabytes);
}

void TranspositionTable::resize(std::size_t megabytes) {
    megabytes = std::max<std::size_t>(1, megabytes);
    const std::size_t requested = std::max<std::size_t>(1,
        (megabytes * BYTES_PER_MB) / sizeof(Cluster));

    std::size_t clusters = 1;
    while (clusters <= requested / 2)
        clusters <<= 1;

    table_ = std::make_unique<Cluster[]>(clusters);
    cluster_count_ = clusters;
    cluster_mask_ = clusters - 1;
    megabytes_ = std::max<std::size_t>(1, (clusters * sizeof(Cluster)) / BYTES_PER_MB);
    clear();
}

void TranspositionTable::clear() {
    generation_ = 0;
    if (table_)
        std::fill_n(table_.get(), cluster_count_, Cluster{});
}

void TranspositionTable::new_search() {
    generation_ = std::uint8_t((generation_ + 1) & GENERATION_MASK);
}

std::uint16_t TranspositionTable::pack_move(Move move) {
    if (!move.isOk())
        return 0;

    const std::uint16_t promotion = move.isPromotion()
        ? std::uint16_t(move.promotionPiece & 0x0Fu) : 0;
    return std::uint16_t(move.from_sq()
                       | (std::uint16_t(move.to_sq()) << 6)
                       | (promotion << 12));
}

Move TranspositionTable::unpack_move(std::uint16_t raw) {
    Move move;
    if (raw == 0)
        return move;

    move.from = Square(raw & 0x3Fu);
    move.to = Square((raw >> 6) & 0x3Fu);
    const std::uint8_t promotion = std::uint8_t((raw >> 12) & 0x0Fu);
    if (promotion != 0) {
        move.flags = Move::PROMOTION;
        move.promotionPiece = Piece(promotion);
    }
    return move;
}

std::uint8_t TranspositionTable::pack_meta(std::uint8_t generation, Bound bound, bool pv) {
    return std::uint8_t((generation & GENERATION_MASK)
                      | (std::uint8_t(bound) << 5)
                      | (pv ? PV_MASK : 0));
}

std::uint8_t TranspositionTable::relative_age(std::uint8_t current, std::uint8_t stored) {
    return std::uint8_t((current - stored) & GENERATION_MASK);
}

TranspositionTable::Cluster::Entry*
TranspositionTable::replacement_slot(Cluster& c, Key key) const {
    Cluster::Entry* replace = &c.entry[0];
    int lowestScore = std::numeric_limits<int>::max();

    for (auto& entry : c.entry) {
        if (!entry.occupied() || entry.matches(key))
            return &entry;

        const int score = int(entry.search_depth())
                        - 4 * int(relative_age(generation_, entry.generation()));
        if (score < lowestScore) {
            lowestScore = score;
            replace = &entry;
        }
    }

    return replace;
}

TTData TranspositionTable::probe(Key key) const {
    const Cluster& c = cluster(key);
    for (const auto& entry : c.entry) {
        if (entry.occupied() && entry.matches(key)) {
            TTData data;
            data.move = unpack_move(entry.move);
            data.value = entry.value;
            data.depth = entry.search_depth();
            data.bound = entry.bound();
            data.pv = entry.pv();
            data.hit = true;
            return data;
        }
    }
    return {};
}

void TranspositionTable::store(Key key, Value value, Bound bound, Depth depth, Move move, bool pv) {
    Cluster& c = cluster(key);
    Cluster::Entry* entry = replacement_slot(c, key);

    if (entry->matches(key) && !move.isOk())
        move = unpack_move(entry->move);

    const bool replace = !entry->occupied()
                      || !entry->matches(key)
                      || bound == BOUND_EXACT
                      || depth >= entry->search_depth()
                      || relative_age(generation_, entry->generation()) != 0;
    if (!replace)
        return;

    entry->keyLo = std::uint32_t(key);
    entry->keyHi = std::uint32_t(key >> 32);
    entry->move = pack_move(move);
    entry->value = value;
    entry->depth = std::uint8_t(std::clamp(depth, 0, 255));
    entry->meta = pack_meta(generation_, bound, pv);
}

std::uint32_t TranspositionTable::hashfull() const {
    if (!table_)
        return 0;

    constexpr std::size_t SAMPLE_CLUSTERS = 1000;
    const std::size_t samples = std::min(cluster_count_, SAMPLE_CLUSTERS);
    std::size_t occupied = 0;

    for (std::size_t i = 0; i < samples; ++i) {
        for (const auto& entry : table_[i].entry)
            occupied += entry.occupied()
                     && relative_age(generation_, entry.generation()) == 0;
    }

    return std::uint32_t((occupied * 1000) /
                         std::max<std::size_t>(1, samples * CLUSTER_SIZE));
}

Value TranspositionTable::value_to_tt(Value value, int ply) {
    if (value >= VALUE_MATE - MAX_PLY)
        return value + ply;
    if (value <= -VALUE_MATE + MAX_PLY)
        return value - ply;
    return value;
}

Value TranspositionTable::value_from_tt(Value value, int ply) {
    if (value >= VALUE_MATE - MAX_PLY)
        return value - ply;
    if (value <= -VALUE_MATE + MAX_PLY)
        return value + ply;
    return value;
}

} // namespace Zero::Search
