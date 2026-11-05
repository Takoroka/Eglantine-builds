// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

#include "common/common_types.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage

    struct Slot {
        Ref ref{};
        u64 tick{};
        u64 device_addr{~u64{0}}; ///< Where the slot's last bind read from
        u32 size{}; // 0: no bytes kept (nothing streamed yet, or bound from a new address)
        u32 bound_size{};  ///< Size of the slot's last stream bind
        u64 bind_serial{}; ///< Draw (KeepWindow::serial) of the slot's last stream bind
        const u8* mapped{}; ///< The stream copy `ref` points at (verify mode)
        std::array<u8, MaxBytes> bytes{};
    };

    /// Uniform slot keep (A/B bit UniformSlotKeep): what the rasterizer saw since the last draw
    /// that bound uniform buffers.
    struct KeepWindow {
        u64 serial{};    ///< Draws that bound uniform buffers, this one included
        u64 keep_from{}; ///< Slots bound at this draw may be kept (0: none, the window is not clean)
        std::span<const std::pair<u64, u64>> written; ///< Guest writes since then (address, size)
        bool verify{};
    };

    struct KeepCounters {
        u64 binds{};       ///< Stream binds through the reuse path, kept ones included
        u64 kept{};        ///< Bound again from the previous draw's copy
        u64 not_clean{};   ///< Binds in draws whose window was not clean
        u64 other_slot{};  ///< Slot last bound before the previous draw, at another address or size
        u64 new_tick{};    ///< The command buffer changed
        u64 written{};     ///< A guest write overlaps the slot
        u64 mismatches{};  ///< Verify mode: kept bytes differ from guest memory (bound normally)
    };

    void Reset() noexcept {
        for (auto& stage : slots) {
            for (Slot& slot : stage) {
                slot.size = 0;
                slot.tick = 0;
                slot.device_addr = ~u64{0};
                slot.bound_size = 0;
                slot.bind_serial = 0;
                slot.mapped = nullptr;
            }
        }
        keep = {};
    }

    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    KeepWindow keep{};
    KeepCounters counters{};
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, reusing the slot's previous allocation when the bytes and the command buffer are the
/// same. Returns false (nothing bound) when the buffer does not qualify, so the caller binds it
/// the classic way.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           size_t stage, u32 index, u32 binding_index,
                                           u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    // Compare against guest memory in place when the range is contiguous on the host and copy it
    // only when it changed: most binds are the same bytes again, and reading every buffer into a
    // scratch copy first was most of this path's cost.
    const u8* source = nullptr;
    if constexpr (requires { device_memory.GetSpan(device_addr, size_t{size}); }) {
        source = device_memory.GetSpan(device_addr, size);
    }
    if (source == nullptr) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        source = reuse.scratch.data();
    }

    if (slot.device_addr != device_addr) {
        // Bound from a new address: games stream each draw's uniforms to a new place in a ring,
        // so the bytes are rarely the last bind's. Copy them to the stream once, without comparing
        // or keeping them; the slot keeps bytes again once binds repeat an address.
        slot.device_addr = device_addr;
        slot.size = 0;
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), source, size);
        slot.ref = runtime.LastStreamUniformRef();
        slot.tick = runtime.CurrentTick();
        slot.bound_size = size;
        slot.bind_serial = reuse.keep.serial;
        slot.mapped = span.data();
        ++reuse.counters.binds;
        return true;
    }
    ++reuse.counters.binds;
    slot.device_addr = device_addr;
    slot.bound_size = size;
    slot.bind_serial = reuse.keep.serial;
    const bool same_bytes =
        slot.size == size && std::memcmp(slot.bytes.data(), source, size) == 0;
    if (same_bytes && slot.tick == runtime.CurrentTick()) {
        runtime.RebindStreamUniform(slot.ref, size);
        return true;
    }
    if (!same_bytes) {
        // Through the slot: the stream gets exactly the bytes later binds compare against, even
        // when the guest writes the buffer meanwhile.
        slot.size = size;
        std::memcpy(slot.bytes.data(), source, size);
    }
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    std::memcpy(span.data(), slot.bytes.data(), size);
    slot.ref = runtime.LastStreamUniformRef();
    // After the allocation: the tick the stream region is tagged with is the one current now.
    slot.tick = runtime.CurrentTick();
    slot.mapped = span.data();
    return true;
}

/// Starts a draw's uniform buffer binds: `clean` when nothing but the guest writes in `written`
/// happened since the previous draw that bound uniform buffers (the rasterizer's keep window).
template <class Runtime>
inline void BeginUniformKeep(bool clean, std::span<const std::pair<u64, u64>> written,
                             bool verify) noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        auto& keep = UniformStreamReuseState<typename Runtime::StreamUniformRef>().keep;
        ++keep.serial;
        keep.keep_from = clean ? keep.serial - 1 : 0;
        keep.written = written;
        keep.verify = verify;
    }
}

/// Ends the draw's binds: binds outside a draw's window never keep.
template <class Runtime>
inline void EndUniformKeep() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        auto& keep = UniformStreamReuseState<typename Runtime::StreamUniformRef>().keep;
        keep.keep_from = 0;
        keep.written = {};
    }
}

/// Binds (stage, cbuf `index`) again from the stream copy its bind at the previous uniform-binding
/// draw made, when that bind read the same address and size, the command buffer is the same and
/// no guest write since overlaps it (with a clean window, nothing else could have changed those
/// bytes: CPU writes, other engines and recorded work end the window). Saves reading guest memory,
/// the compare and the GPU-modified lookup. Returns false when the caller binds it as usual.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool KeepSmallUniform(Runtime& runtime, DeviceMemory& device_memory, size_t stage,
                                    u32 index, u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (stage >= Reuse::NumStages || index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    auto& counters = reuse.counters;
    if (reuse.keep.keep_from == 0) {
        ++counters.not_clean;
        return false;
    }
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    if (slot.bind_serial != reuse.keep.keep_from || slot.device_addr != device_addr ||
        slot.bound_size != size || size == 0) {
        ++counters.other_slot;
        return false;
    }
    if (slot.tick != runtime.CurrentTick()) {
        ++counters.new_tick;
        return false;
    }
    const u64 end = device_addr + size;
    for (const auto& [addr, length] : reuse.keep.written) {
        if (addr < end && device_addr < addr + length) {
            ++counters.written;
            return false;
        }
    }
    if (reuse.keep.verify && slot.mapped != nullptr) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        if (std::memcmp(reuse.scratch.data(), slot.mapped, size) != 0) {
            ++counters.mismatches;
            return false;
        }
    }
    runtime.RebindStreamUniform(slot.ref, size);
    slot.bind_serial = reuse.keep.serial;
    ++counters.binds;
    ++counters.kept;
    return true;
}

template <class Runtime>
[[nodiscard]] inline auto UniformKeepCounters() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        return UniformStreamReuseState<typename Runtime::StreamUniformRef>().counters;
    } else {
        return 0;
    }
}

} // namespace VideoCommon
