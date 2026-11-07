// SPDX-License-Identifier: GPL-3.0-or-later

#include <tuple>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/nx2_method_tally.h"

namespace {

using Tegra::CommandHeader;
using Tegra::MethodHistogram;
using Tegra::MethodTallyState;
using Tegra::SubmissionMode;
using Run = std::tuple<u32, u32, u32>; // subchannel, method, words

CommandHeader Header(SubmissionMode mode, u32 subchannel, u32 method, u32 count) {
    CommandHeader header{};
    header.method.Assign(method);
    header.subchannel.Assign(subchannel);
    header.method_count.Assign(count);
    header.mode.Assign(mode);
    return header;
}

CommandHeader Data(u32 value) {
    CommandHeader word{};
    word.argument = value;
    return word;
}

constexpr MethodTallyState Clean{0, 0, 0, false, false};

std::vector<Run> Tally(const std::vector<CommandHeader>& words, MethodTallyState state = Clean) {
    std::vector<Run> runs;
    Tegra::ForEachMethodWords(words, state, [&](u32 subchannel, u32 method, u32 count) {
        runs.emplace_back(subchannel, method, count);
    });
    return runs;
}

} // Anonymous namespace

TEST_CASE("Increasing methods get one word each", "[video_core][method_tally]") {
    const auto runs = Tally({Header(SubmissionMode::Increasing, 1, 0x100, 3), Data(1), Data(2),
                             Data(3)});
    REQUIRE(runs == std::vector<Run>{{1, 0x100, 1}, {1, 0x101, 1}, {1, 0x102, 1}});
}

TEST_CASE("Non-increasing data goes to one method", "[video_core][method_tally]") {
    // A macro's five parameters, then an increasing pair.
    const auto runs = Tally({Header(SubmissionMode::NonIncreasing, 0, 0xe51, 5), Data(1), Data(2),
                             Data(3), Data(4), Data(5),
                             Header(SubmissionMode::Increasing, 0, 0x200, 2), Data(6), Data(7)});
    REQUIRE(runs == std::vector<Run>{{0, 0xe51, 5}, {0, 0x200, 1}, {0, 0x201, 1}});
}

TEST_CASE("Increase-once data moves on once", "[video_core][method_tally]") {
    const auto runs = Tally({Header(SubmissionMode::IncreaseOnce, 2, 0x300, 4), Data(1), Data(2),
                             Data(3), Data(4)});
    REQUIRE(runs == std::vector<Run>{{2, 0x300, 1}, {2, 0x301, 3}});
}

TEST_CASE("Inline methods carry their argument in the header", "[video_core][method_tally]") {
    const auto runs = Tally({Header(SubmissionMode::Inline, 3, 0x1d, 0x123),
                             Header(SubmissionMode::Inline, 3, 0x1e, 0x7)});
    REQUIRE(runs == std::vector<Run>{{3, 0x1d, 1}, {3, 0x1e, 1}});
}

TEST_CASE("An entry continues the method the last one left", "[video_core][method_tally]") {
    // Three parameters left of a macro call, then a new header.
    const MethodTallyState left{0xe11, 0, 3, true, false};
    const auto runs = Tally({Data(1), Data(2), Data(3),
                             Header(SubmissionMode::Increasing, 1, 0x10, 1), Data(4)},
                            left);
    REQUIRE(runs == std::vector<Run>{{0, 0xe11, 3}, {1, 0x10, 1}});
    // An entry shorter than what is left: its words only.
    const MethodTallyState longer{0xe11, 0, 10, true, false};
    REQUIRE(Tally({Data(1), Data(2), Data(3), Data(4)}, longer) ==
            std::vector<Run>{{0, 0xe11, 4}});
    // An increasing method left: it goes on increasing.
    const MethodTallyState increasing{0x400, 4, 2, false, false};
    REQUIRE(Tally({Data(1), Data(2)}, increasing) ==
            std::vector<Run>{{4, 0x400, 1}, {4, 0x401, 1}});
}

TEST_CASE("Old submission modes are skipped as ProcessCommands does", "[video_core][method_tally]") {
    const auto runs = Tally({Header(SubmissionMode::IncreasingOld, 0, 0x100, 2),
                             Header(SubmissionMode::Increasing, 0, 0x110, 1), Data(9)});
    REQUIRE(runs == std::vector<Run>{{0, 0x110, 1}});
}

TEST_CASE("The histogram ranks methods by words and starts over", "[video_core][method_tally]") {
    MethodHistogram histogram;
    REQUIRE(histogram.TakeTop(4).empty());
    REQUIRE(MethodHistogram::Format({}, 0, 0) == "0 entries, no words");
    histogram.AddEntry();
    histogram.AddEntry();
    // A macro's start and parameter methods count together.
    histogram.Add(MethodHistogram::Maxwell3D, 0xe50, 1);
    histogram.Add(MethodHistogram::Maxwell3D, 0xe51, 5);
    histogram.Add(MethodHistogram::Compute, 0x2b4, 3);
    histogram.Add(MethodHistogram::Puller, 0x1b, 1);
    histogram.Add(MethodHistogram::Maxwell3D, 0x2b4, 1);
    REQUIRE(histogram.Entries() == 2);
    REQUIRE(histogram.Words() == 11);
    const auto top = histogram.TakeTop(3);
    REQUIRE(top.size() == 3);
    REQUIRE(top[0].kind == MethodHistogram::Maxwell3D);
    REQUIRE(top[0].method == 0xe50);
    REQUIRE(top[0].words == 6);
    REQUIRE(top[1].kind == MethodHistogram::Compute);
    REQUIRE(top[1].words == 3);
    // Ties: by kind, then method.
    REQUIRE(top[2].kind == MethodHistogram::Puller);
    REQUIRE(MethodHistogram::Format(top, 2, 11) ==
            "2 entries, 11 words: 3D macro 0xe50 55% compute 0x2b4 27% puller 0x1b 9%");
    REQUIRE(histogram.Words() == 0);
    REQUIRE(histogram.Entries() == 0);
    REQUIRE(histogram.TakeTop(3).empty());
}
