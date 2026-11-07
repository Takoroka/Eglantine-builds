// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace Shader::Optimization {

/// Tarjan's strongly connected components with an explicit stack of frames, not recursion.
///
/// A recursive walk takes one native frame per node of the deepest path, and a native program's
/// add, phi and select chains can be thousands of instructions deep: on 11-07 Kirby Air Riders
/// overflowed the 1 MiB stack of the split GPU thread's back 1647 frames into the storage buffer
/// pass's iterator-word search, and the process died without a crash report.
///
/// Visits nodes and reports components in the same order as the recursive algorithm: children
/// in the order NextChild gives them, a component when its root finishes, its members in the
/// order they leave the stack (the root last).
///
/// next_child(Node node, size_t& cursor, Node& child) -> bool: stores the node's next followed
/// child at or after `cursor` in `child`, moves `cursor` past it and returns true, or returns
/// false when no child is left. `cursor` starts at 0 for each node.
/// on_component(std::span<const Node> members): called once per component, any size.
template <typename Node>
class StronglyConnectedComponents {
public:
    /// Visits the nodes reachable from `root` that no earlier Visit reached.
    template <typename NextChild, typename OnComponent>
    void Visit(Node root, NextChild&& next_child, OnComponent&& on_component) {
        if (infos.contains(root)) {
            return;
        }
        Enter(root);
        while (!frames.empty()) {
            Frame& frame{frames.back()};
            const Node node{frame.node};
            Node child{};
            bool descend{};
            while (next_child(node, frame.cursor, child)) {
                const auto found{infos.find(child)};
                if (found == infos.end()) {
                    descend = true;
                    break;
                }
                if (found->second.on_stack) {
                    Info& info{infos.at(node)};
                    info.lowlink = (std::min)(info.lowlink, found->second.index);
                }
            }
            if (descend) {
                Enter(child); // `frame` is stale from here on
                continue;
            }
            frames.pop_back();
            const Info done{infos.at(node)};
            if (done.lowlink == done.index) {
                component.clear();
                Node member{};
                do {
                    member = stack.back();
                    stack.pop_back();
                    infos.at(member).on_stack = false;
                    component.push_back(member);
                } while (member != node);
                on_component(std::span<const Node>{component});
            }
            if (!frames.empty()) {
                Info& parent{infos.at(frames.back().node)};
                parent.lowlink = (std::min)(parent.lowlink, done.lowlink);
            }
        }
    }

    /// The deepest path walked so far, in nodes: the frames a recursive walk would have needed.
    [[nodiscard]] size_t MaxDepth() const noexcept {
        return max_depth;
    }

private:
    struct Info {
        u32 index;
        u32 lowlink;
        bool on_stack;
    };
    struct Frame {
        Node node;
        size_t cursor;
    };

    void Enter(Node node) {
        const u32 index{next_index++};
        infos.emplace(node, Info{index, index, true});
        stack.push_back(node);
        frames.push_back(Frame{node, 0});
        max_depth = (std::max)(max_depth, frames.size());
    }

    std::unordered_map<Node, Info> infos;
    std::vector<Node> stack;
    std::vector<Frame> frames;
    std::vector<Node> component;
    u32 next_index{};
    size_t max_depth{};
};

} // namespace Shader::Optimization
