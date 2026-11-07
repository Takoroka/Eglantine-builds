// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "video_core/dma_pusher.h"

namespace Tegra {

/// A method's data as it stands where a run of command words starts (DmaPusher's dma_state).
struct MethodTallyState {
    u32 method;
    u32 subchannel;
    u32 method_count;
    bool non_incrementing;
    bool increment_once;
};

/// Calls `tally(subchannel, method, words)` for the data words each method gets in `commands`,
/// decoded as DmaPusher::ProcessCommands does from `state`. Header words count for no method.
template <typename Tally>
void ForEachMethodWords(std::span<const CommandHeader> commands, MethodTallyState state,
                        Tally&& tally) {
    for (size_t index = 0; index < commands.size();) {
        if (state.method_count != 0 && state.non_incrementing) {
            const u32 words =
                static_cast<u32>((std::min)(size_t{state.method_count}, commands.size() - index));
            tally(state.subchannel, state.method, words);
            state.method_count -= words;
            index += words;
        } else if (state.method_count != 0) {
            tally(state.subchannel, state.method, 1U);
            ++state.method;
            state.non_incrementing = state.increment_once;
            --state.method_count;
            ++index;
        } else {
            const CommandHeader header = commands[index];
            switch (header.mode) {
            case SubmissionMode::Increasing:
            case SubmissionMode::NonIncreasing:
            case SubmissionMode::IncreaseOnce:
                state.method = header.method;
                state.subchannel = header.subchannel;
                state.method_count = header.method_count;
                state.non_incrementing = header.mode == SubmissionMode::NonIncreasing;
                state.increment_once = header.mode == SubmissionMode::IncreaseOnce;
                break;
            case SubmissionMode::Inline:
                // The argument is in the header itself.
                state.method = header.method;
                state.subchannel = header.subchannel;
                state.non_incrementing = true;
                state.increment_once = false;
                tally(state.subchannel, state.method, 1U);
                break;
            case SubmissionMode::IncreasingOld:
            case SubmissionMode::NonIncreasingOld:
            default:
                // Skipped, as ProcessCommands does.
                break;
            }
            ++index;
        }
    }
}

/// Words per method in command list entries, summed over a window and reported by share (the
/// GPU-written entries the front still drains for, DmaPusher::FlushGpuWrittenCommands, and those
/// it waits for a copy of). GPU thread only.
class MethodHistogram {
public:
    /// Who runs the method: methods below 0x40 are the puller's whatever the subchannel.
    enum Kind : u32 {
        Puller,
        Maxwell3D,
        Compute,
        Dma,
        Fermi2D,
        InlineToMemory,
        Timer,
        Unbound,
        KindCount,
    };
    static constexpr u32 MethodBits = 13;
    static constexpr u32 MethodMask = (1U << MethodBits) - 1;
    static constexpr u32 MacroRegistersStart = 0xE00;

    struct Item {
        Kind kind;
        u32 method;
        u64 words;
    };

    void Add(Kind kind, u32 method, u32 words) {
        if (counts.empty()) {
            // On the first entry that needs it (titles that never drain keep nothing).
            counts.resize(size_t{KindCount} << MethodBits);
        }
        method &= MethodMask;
        if (kind == Maxwell3D && method >= MacroRegistersStart) {
            // A macro's start method and its parameter method count as one.
            method &= ~1U;
        }
        u32& count = counts[(static_cast<size_t>(kind) << MethodBits) | method];
        count = words > ~count ? ~0U : count + words;
        total_words += words;
    }

    void AddEntry() {
        ++total_entries;
    }

    [[nodiscard]] u64 Words() const {
        return total_words;
    }
    [[nodiscard]] u64 Entries() const {
        return total_entries;
    }

    /// The `count` methods with the most words, most first, and starts a new window.
    std::vector<Item> TakeTop(size_t count) {
        std::vector<Item> items;
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] != 0) {
                items.push_back({static_cast<Kind>(i >> MethodBits),
                                 static_cast<u32>(i & MethodMask), counts[i]});
            }
        }
        const auto more = [](const Item& a, const Item& b) {
            return a.words != b.words ? a.words > b.words
                                      : (a.kind != b.kind ? a.kind < b.kind : a.method < b.method);
        };
        const size_t kept = (std::min)(count, items.size());
        std::partial_sort(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(kept),
                          items.end(), more);
        items.resize(kept);
        std::fill(counts.begin(), counts.end(), 0U);
        total_words = 0;
        total_entries = 0;
        return items;
    }

    [[nodiscard]] static const char* KindName(Kind kind) {
        switch (kind) {
        case Puller:
            return "puller";
        case Maxwell3D:
            return "3D";
        case Compute:
            return "compute";
        case Dma:
            return "DMA";
        case Fermi2D:
            return "2D";
        case InlineToMemory:
            return "inline";
        case Timer:
            return "timer";
        case Unbound:
        case KindCount:
            break;
        }
        return "unbound";
    }

    /// "<entries> entries, <words> words: 3D macro 0xe50 41% ..." for a window's top methods.
    [[nodiscard]] static std::string Format(const std::vector<Item>& top, u64 entries, u64 words) {
        if (words == 0) {
            return fmt::format("{} entries, no words", entries);
        }
        std::string text = fmt::format("{} entries, {} words:", entries, words);
        for (const Item& item : top) {
            const bool macro = item.kind == Maxwell3D && item.method >= MacroRegistersStart;
            text += fmt::format(" {}{} {:#x} {:.0f}%", KindName(item.kind), macro ? " macro" : "",
                                item.method,
                                100.0 * static_cast<double>(item.words) /
                                    static_cast<double>(words));
        }
        return text;
    }

private:
    std::vector<u32> counts; ///< Words per Kind and method (KindCount << MethodBits), once used
    u64 total_words{};
    u64 total_entries{};
};

} // namespace Tegra
