// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/nx2_counters.h"
#include "core/nx2_cpu_stats.h"

namespace Core::Nx2CpuStats {

namespace {

constexpr size_t NUM_CORES = Hardware::NUM_CPU_CORES;
constexpr size_t TOP_SVCS = 12;

/// Names of the SVC numbers (Kernel::Svc::SvcId, plus the Switch 2 one Svc::Call handles).
const char* SvcName(size_t id) {
    switch (id) {
    case 0x01: return "SetHeapSize";
    case 0x02: return "SetMemoryPermission";
    case 0x03: return "SetMemoryAttribute";
    case 0x04: return "MapMemory";
    case 0x05: return "UnmapMemory";
    case 0x06: return "QueryMemory";
    case 0x07: return "ExitProcess";
    case 0x08: return "CreateThread";
    case 0x09: return "StartThread";
    case 0x0a: return "ExitThread";
    case 0x0b: return "SleepThread";
    case 0x0c: return "GetThreadPriority";
    case 0x0d: return "SetThreadPriority";
    case 0x0e: return "GetThreadCoreMask";
    case 0x0f: return "SetThreadCoreMask";
    case 0x10: return "GetCurrentProcessorNumber";
    case 0x11: return "SignalEvent";
    case 0x12: return "ClearEvent";
    case 0x13: return "MapSharedMemory";
    case 0x14: return "UnmapSharedMemory";
    case 0x15: return "CreateTransferMemory";
    case 0x16: return "CloseHandle";
    case 0x17: return "ResetSignal";
    case 0x18: return "WaitSynchronization";
    case 0x19: return "CancelSynchronization";
    case 0x1a: return "ArbitrateLock";
    case 0x1b: return "ArbitrateUnlock";
    case 0x1c: return "WaitProcessWideKeyAtomic";
    case 0x1d: return "SignalProcessWideKey";
    case 0x1e: return "GetSystemTick";
    case 0x1f: return "ConnectToNamedPort";
    case 0x20: return "SendSyncRequestLight";
    case 0x21: return "SendSyncRequest";
    case 0x22: return "SendSyncRequestWithUserBuffer";
    case 0x23: return "SendAsyncRequestWithUserBuffer";
    case 0x24: return "GetProcessId";
    case 0x25: return "GetThreadId";
    case 0x26: return "Break";
    case 0x27: return "OutputDebugString";
    case 0x28: return "ReturnFromException";
    case 0x29: return "GetInfo";
    case 0x2a: return "FlushEntireDataCache";
    case 0x2b: return "FlushDataCache";
    case 0x2c: return "MapPhysicalMemory";
    case 0x2d: return "UnmapPhysicalMemory";
    case 0x2e: return "GetDebugFutureThreadInfo";
    case 0x2f: return "GetLastThreadInfo";
    case 0x30: return "GetResourceLimitLimitValue";
    case 0x31: return "GetResourceLimitCurrentValue";
    case 0x32: return "SetThreadActivity";
    case 0x33: return "GetThreadContext3";
    case 0x34: return "WaitForAddress";
    case 0x35: return "SignalToAddress";
    case 0x36: return "SynchronizePreemptionState";
    case 0x37: return "GetResourceLimitPeakValue";
    case 0x39: return "CreateIoPool";
    case 0x3a: return "CreateIoRegion";
    case 0x3c: return "KernelDebug";
    case 0x3d: return "ChangeKernelTraceState";
    case 0x40: return "CreateSession";
    case 0x41: return "AcceptSession";
    case 0x42: return "ReplyAndReceiveLight";
    case 0x43: return "ReplyAndReceive";
    case 0x44: return "ReplyAndReceiveWithUserBuffer";
    case 0x45: return "CreateEvent";
    case 0x46: return "MapIoRegion";
    case 0x47: return "UnmapIoRegion";
    case 0x48: return "MapPhysicalMemoryUnsafe";
    case 0x49: return "UnmapPhysicalMemoryUnsafe";
    case 0x4a: return "SetUnsafeLimit";
    case 0x4b: return "CreateCodeMemory";
    case 0x4c: return "ControlCodeMemory";
    case 0x4d: return "SleepSystem";
    case 0x4e: return "ReadWriteRegister";
    case 0x4f: return "SetProcessActivity";
    case 0x50: return "CreateSharedMemory";
    case 0x51: return "MapTransferMemory";
    case 0x52: return "UnmapTransferMemory";
    case 0x53: return "CreateInterruptEvent";
    case 0x54: return "QueryPhysicalAddress";
    case 0x55: return "QueryIoMapping";
    case 0x56: return "CreateDeviceAddressSpace";
    case 0x57: return "AttachDeviceAddressSpace";
    case 0x58: return "DetachDeviceAddressSpace";
    case 0x59: return "MapDeviceAddressSpaceByForce";
    case 0x5a: return "MapDeviceAddressSpaceAligned";
    case 0x5c: return "UnmapDeviceAddressSpace";
    case 0x5d: return "InvalidateProcessDataCache";
    case 0x5e: return "StoreProcessDataCache";
    case 0x5f: return "FlushProcessDataCache";
    case 0x60: return "DebugActiveProcess";
    case 0x61: return "BreakDebugProcess";
    case 0x62: return "TerminateDebugProcess";
    case 0x63: return "GetDebugEvent";
    case 0x64: return "ContinueDebugEvent";
    case 0x65: return "GetProcessList";
    case 0x66: return "GetThreadList";
    case 0x67: return "GetDebugThreadContext";
    case 0x68: return "SetDebugThreadContext";
    case 0x69: return "QueryDebugProcessMemory";
    case 0x6a: return "ReadDebugProcessMemory";
    case 0x6b: return "WriteDebugProcessMemory";
    case 0x6c: return "SetHardwareBreakPoint";
    case 0x6d: return "GetDebugThreadParam";
    case 0x6f: return "GetSystemInfo";
    case 0x70: return "CreatePort";
    case 0x71: return "ManageNamedPort";
    case 0x72: return "ConnectToPort";
    case 0x73: return "SetProcessMemoryPermission";
    case 0x74: return "MapProcessMemory";
    case 0x75: return "UnmapProcessMemory";
    case 0x76: return "QueryProcessMemory";
    case 0x77: return "MapProcessCodeMemory";
    case 0x78: return "UnmapProcessCodeMemory";
    case 0x79: return "CreateProcess";
    case 0x7a: return "StartProcess";
    case 0x7b: return "TerminateProcess";
    case 0x7c: return "GetProcessInfo";
    case 0x7d: return "CreateResourceLimit";
    case 0x7e: return "SetResourceLimitLimitValue";
    case 0x7f: return "CallSecureMonitor";
    case 0x90: return "MapInsecureMemory";
    case 0x91: return "UnmapInsecureMemory";
    case 0x80: return "SetGpuMemoryAttribute";
    default: return "?";
    }
}

constexpr std::array<const char*, static_cast<size_t>(Hint::Count)> HINT_NAMES{
    "WFE", "WFI", "YIELD", "SEV", "SEVL"};

} // Anonymous namespace

void Report(s64 window_ns) {
    using namespace Common::Nx2Counters;

    // SVCs: the top ids over all cores, each with its per-core rates.
    std::array<std::array<u64, NUM_CORES>, NUM_SVC_IDS> svc_per_core{};
    std::array<u64, NUM_SVC_IDS> svc_totals{};
    std::array<u64, NUM_CORES> core_totals{};
    u64 svc_total = 0;
    for (size_t core = 0; core < NUM_CORES; ++core) {
        for (size_t id = 0; id < NUM_SVC_IDS; ++id) {
            const u64 count = g_cores[core].svcs[id].exchange(0, std::memory_order_relaxed);
            svc_per_core[id][core] = count;
            svc_totals[id] += count;
            core_totals[core] += count;
            svc_total += count;
        }
    }
    std::string top;
    for (const size_t id : TopIndices(svc_totals, TOP_SVCS)) {
        top += fmt::format("{}{:#04x} {} {} ({})", top.empty() ? "" : ", ", id, SvcName(id),
                           FormatRate(PerSecond(svc_totals[id], window_ns)),
                           FormatPerCore(svc_per_core[id], window_ns));
    }
    LOG_INFO(Kernel_SVC,
             "nx2 SVC mix (last {:.1f} s): {}/s, per core {} | top {} by calls/s (per core): {}",
             static_cast<double>(window_ns) / 1e9, FormatRate(PerSecond(svc_total, window_ns)),
             FormatPerCore(core_totals, window_ns), TOP_SVCS, top.empty() ? "none" : top);

    // Hints: every kind, with its per-core rates.
    std::string hints;
    u64 hint_total = 0;
    for (size_t hint = 0; hint < HINT_NAMES.size(); ++hint) {
        std::array<u64, NUM_CORES> per_core{};
        u64 total = 0;
        for (size_t core = 0; core < NUM_CORES; ++core) {
            per_core[core] = g_cores[core].hints[hint].exchange(0, std::memory_order_relaxed);
            total += per_core[core];
        }
        hint_total += total;
        hints += fmt::format(" | {} {}/s ({})", HINT_NAMES[hint],
                             FormatRate(PerSecond(total, window_ns)),
                             FormatPerCore(per_core, window_ns));
    }
    LOG_INFO(Core_ARM, "nx2 CPU hints (last {:.1f} s, per core in brackets): {}/s{}",
             static_cast<double>(window_ns) / 1e9, FormatRate(PerSecond(hint_total, window_ns)),
             hints);
}

} // namespace Core::Nx2CpuStats
