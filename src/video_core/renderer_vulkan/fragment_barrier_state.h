// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Vulkan {

/// A fragment barrier the guest asked for while a render pass was active, not recorded yet
/// (A/B bit FragmentBarrierDefer). Ending a render pass records a barrier from the attachment
/// writes to every later read (Scheduler::EndRenderPass), which is all the guest's barrier asks
/// for, so the pass may go on until something reads what it wrote.
class FragmentBarrierState {
public:
    /// The guest asked for a fragment barrier with a pass active.
    void Defer() noexcept {
        pending = true;
    }

    [[nodiscard]] bool Pending() const noexcept {
        return pending;
    }

    /// A render pass is requested. keeps: the caller knows what follows samples none of the
    /// pass's attachments (a draw whose setup checked, a clear). Others (blits, texture draws)
    /// may sample one: the pass ends first.
    [[nodiscard]] bool EndsPassOnRequest(bool keeps) const noexcept {
        return pending && !keeps;
    }

    /// A draw's full setup: the pass ends before a draw that samples one of its render targets.
    /// Any other draw leaves the barrier pending for the draws after it.
    [[nodiscard]] bool EndsPassForDraw(bool samples_render_target) const noexcept {
        return pending && samples_render_target;
    }

    /// The pass ended (for whatever reason): its end recorded the barrier.
    void OnPassEnd() noexcept {
        pending = false;
    }

private:
    bool pending = false;
};

} // namespace Vulkan
