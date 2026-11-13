// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "shader_recompiler/ir_opt/strongly_connected.h"

using Shader::Optimization::StronglyConnectedComponents;

namespace {
using Graph = std::vector<std::vector<int>>;
using Components = std::vector<std::vector<int>>;

/// The recursive Tarjan the storage buffer pass's iterator-word search used before this walker.
struct RecursiveTarjan {
    explicit RecursiveTarjan(const Graph& graph_) : graph{graph_} {}

    void Visit(int node) {
        if (indices.contains(node)) {
            return;
        }
        const u32 index = next_index++;
        indices[node] = lowlinks[node] = index;
        stack.push_back(node);
        on_stack.insert(node);
        for (const int child : graph[node]) {
            if (!indices.contains(child)) {
                Visit(child);
                lowlinks[node] = (std::min)(lowlinks[node], lowlinks[child]);
            } else if (on_stack.contains(child)) {
                lowlinks[node] = (std::min)(lowlinks[node], indices[child]);
            }
        }
        if (lowlinks[node] != index) {
            return;
        }
        std::vector<int> component;
        int member;
        do {
            member = stack.back();
            stack.pop_back();
            on_stack.erase(member);
            component.push_back(member);
        } while (member != node);
        components.push_back(std::move(component));
    }

    const Graph& graph;
    std::unordered_map<int, u32> indices;
    std::unordered_map<int, u32> lowlinks;
    std::unordered_set<int> on_stack;
    std::vector<int> stack;
    u32 next_index{};
    Components components;
};

Components Iterative(const Graph& graph, std::span<const int> roots, size_t* max_depth = nullptr) {
    StronglyConnectedComponents<int> scc;
    Components components;
    const auto next_child = [&](int node, size_t& cursor, int& child) {
        const auto& edges{graph[static_cast<size_t>(node)]};
        if (cursor >= edges.size()) {
            return false;
        }
        child = edges[cursor++];
        return true;
    };
    const auto on_component = [&](std::span<const int> members) {
        components.emplace_back(members.begin(), members.end());
    };
    for (const int root : roots) {
        scc.Visit(root, next_child, on_component);
    }
    if (max_depth) {
        *max_depth = scc.MaxDepth();
    }
    return components;
}

Components Recursive(const Graph& graph, std::span<const int> roots) {
    RecursiveTarjan tarjan{graph};
    for (const int root : roots) {
        tarjan.Visit(root);
    }
    return tarjan.components;
}
} // Anonymous namespace

TEST_CASE("Components match the recursive Tarjan, in the same order", "[video_core][scc]") {
    std::mt19937 rng{0x5cc1107};
    for (int round = 0; round < 2000; ++round) {
        const int nodes = std::uniform_int_distribution<int>{1, 60}(rng);
        const int max_edges = std::uniform_int_distribution<int>{0, 4}(rng);
        Graph graph(static_cast<size_t>(nodes));
        std::uniform_int_distribution<int> pick{0, nodes - 1};
        for (auto& edges : graph) {
            // Self loops and repeated edges included, as phis can name the same value twice.
            const int count = std::uniform_int_distribution<int>{0, max_edges}(rng);
            for (int edge = 0; edge < count; ++edge) {
                edges.push_back(pick(rng));
            }
        }
        std::vector<int> roots;
        const int root_count = std::uniform_int_distribution<int>{1, nodes}(rng);
        for (int root = 0; root < root_count; ++root) {
            roots.push_back(pick(rng)); // Repeated roots included
        }
        INFO("round " << round);
        REQUIRE(Iterative(graph, roots) == Recursive(graph, roots));
    }
}

TEST_CASE("A hand-checked graph", "[video_core][scc]") {
    // 0 -> 1 -> 2 -> 0 is a cycle, 2 -> 3 -> 4 -> 3 another, 5 stands alone.
    const Graph graph{{1}, {2}, {0, 3}, {4}, {3}, {}};
    const std::vector<int> roots{0, 5, 2};
    size_t depth{};
    const Components components{Iterative(graph, roots, &depth)};
    REQUIRE(components == Components{{4, 3}, {2, 1, 0}, {5}});
    REQUIRE(depth == 5);
}

TEST_CASE("Chains far deeper than a thread stack", "[video_core][scc]") {
    // The recursion took ~600 bytes a level: 200k levels is ~115 MiB, past any thread's stack.
    constexpr int Length = 200'000;
    Graph chain(Length);
    for (int node = 0; node + 1 < Length; ++node) {
        chain[static_cast<size_t>(node)].push_back(node + 1);
    }
    const std::vector<int> roots{0};
    size_t depth{};
    Components components{Iterative(chain, roots, &depth)};
    REQUIRE(depth == Length);
    REQUIRE(components.size() == Length);
    REQUIRE(components.front() == std::vector<int>{Length - 1});
    REQUIRE(components.back() == std::vector<int>{0});

    // Closed into one cycle: a single component, the root last.
    chain.back().push_back(0);
    components = Iterative(chain, roots, &depth);
    REQUIRE(depth == Length);
    REQUIRE(components.size() == 1);
    REQUIRE(components[0].size() == Length);
    REQUIRE(components[0].front() == Length - 1);
    REQUIRE(components[0].back() == 0);
}

TEST_CASE("Later visits skip what earlier ones reached", "[video_core][scc]") {
    const Graph graph{{1}, {0}, {0}};
    StronglyConnectedComponents<int> scc;
    size_t calls{};
    const auto next_child = [&](int node, size_t& cursor, int& child) {
        const auto& edges{graph[static_cast<size_t>(node)]};
        if (cursor >= edges.size()) {
            return false;
        }
        child = edges[cursor++];
        return true;
    };
    const auto count = [&](std::span<const int>) { ++calls; };
    scc.Visit(0, next_child, count);
    REQUIRE(calls == 1);
    scc.Visit(1, next_child, count);
    REQUIRE(calls == 1);
    scc.Visit(2, next_child, count);
    REQUIRE(calls == 2);
}
