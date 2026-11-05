// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <type_traits>
#include <queue>
#include <unordered_set>

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::IR {

template <typename Pred, typename Resolve = std::identity>
auto BreadthFirstSearch(Value value, Pred&& pred, Resolve&& resolve = {})
    -> std::invoke_result_t<Pred, const Inst*> {
    value = resolve(value);
    if (value.IsImmediate()) {
        // Nothing to do with immediates
        return std::nullopt;
    }
    // Breadth-first search visiting the right most arguments first
    // Small vector has been determined from shaders in Super Smash Bros. Ultimate
    boost::container::small_vector<const Inst*, 2> visited;
    // Once a search grows, a linear scan of `visited` per argument made it quadratic: a 28,000
    // instruction Switch 2 compute shader took 3.4 s to translate, most of it here. Same visit
    // order, so the same result.
    std::unordered_set<const Inst*> visited_set;
    const auto first_visit = [&](const Inst* inst) {
        if (!visited_set.empty()) {
            return visited_set.insert(inst).second;
        }
        if (std::ranges::find(visited, inst) != visited.end()) {
            return false;
        }
        visited.push_back(inst);
        if (visited.size() > 64) {
            visited_set.insert(visited.begin(), visited.end());
        }
        return true;
    };
    std::queue<const Inst*> queue;
    queue.push(value.InstRecursive());

    while (!queue.empty()) {
        // Pop one instruction from the queue
        const Inst* const inst{queue.front()};
        queue.pop();
        if (const std::optional result = pred(inst)) {
            // This is the instruction we were looking for
            return result;
        }
        // Visit the right most arguments first
        for (size_t arg = inst->NumArgs(); arg--;) {
            const Value arg_value{resolve(inst->Arg(arg))};
            if (arg_value.IsImmediate()) {
                continue;
            }
            // Queue instruction if it hasn't been visited
            const Inst* const arg_inst{arg_value.InstRecursive()};
            if (first_visit(arg_inst)) {
                queue.push(arg_inst);
            }
        }
    }
    // SSA tree has been traversed and the result hasn't been found
    return std::nullopt;
}

/// BreadthFirstSearch for many searches with the same predicate over an IR that doesn't change
/// meanwhile. `misses` holds instructions from which no match can be reached: they are not
/// searched, and a search that finds nothing adds every instruction it visited. The results are
/// the ones BreadthFirstSearch gives (a skipped instruction is on no path to a match), as long as
/// every search sharing `misses` uses the same predicate and `resolve`.
template <typename Pred, typename Resolve = std::identity>
auto BreadthFirstSearch(Value value, Pred&& pred, std::unordered_set<const Inst*>& misses,
                        Resolve&& resolve = {}) -> std::invoke_result_t<Pred, const Inst*> {
    value = resolve(value);
    if (value.IsImmediate()) {
        return std::nullopt;
    }
    const Inst* const start{value.InstRecursive()};
    if (misses.contains(start)) {
        return std::nullopt;
    }
    // A Switch 2 compute shader made 725 searches that each found nothing after visiting the
    // same ~5,600 instructions: 1.6 s of its 1.8 s translation.
    std::unordered_set<const Inst*> visited;
    std::queue<const Inst*> queue;
    queue.push(start);
    while (!queue.empty()) {
        const Inst* const inst{queue.front()};
        queue.pop();
        if (const std::optional result = pred(inst)) {
            return result;
        }
        // Visit the right most arguments first
        for (size_t arg = inst->NumArgs(); arg--;) {
            const Value arg_value{resolve(inst->Arg(arg))};
            if (arg_value.IsImmediate()) {
                continue;
            }
            const Inst* const arg_inst{arg_value.InstRecursive()};
            if (!misses.contains(arg_inst) && visited.insert(arg_inst).second) {
                queue.push(arg_inst);
            }
        }
    }
    misses.insert(start);
    misses.insert(visited.begin(), visited.end());
    return std::nullopt;
}

} // namespace Shader::IR
