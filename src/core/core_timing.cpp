// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include "common/cpu_features.h"
#include "common/cpu_features.h"
#include "common/nx2_ab.h"

#ifdef _WIN32
#include "common/windows/timer_resolution.h"
#endif

#include "common/logging.h"
#include "common/nx2_counters.h"
#include "common/settings.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include "core/nx2_cpu_stats.h"

namespace Core::Timing {

constexpr s64 MAX_SLICE_LENGTH = 10000;

namespace {

/// Every event type CreateEvent made, for the "nx2 timing" line's per-type counts.
struct EventTypeRegistry {
    std::mutex mutex;
    std::vector<std::weak_ptr<EventType>> types;
};

EventTypeRegistry& EventTypes() {
    // Never destroyed: event types may be made and dropped while statics go away at exit.
    static EventTypeRegistry* const registry = new EventTypeRegistry{};
    return *registry;
}

} // Anonymous namespace

constexpr u32 CPU_CLOCK_BASE_MHZ = 1020;
constexpr u32 CPU_CLOCK_BOOST_MHZ = 1734;
constexpr u32 CPU_CLOCK_OVERCLOCK_MHZ = 2040;

constexpr u32 CpuClockTargetMhz(Settings::CpuClock clock) {
    switch (clock) {
    case Settings::CpuClock::Boost:
        return CPU_CLOCK_BOOST_MHZ;
    case Settings::CpuClock::Overclock:
        return CPU_CLOCK_OVERCLOCK_MHZ;
    default:
        return CPU_CLOCK_BASE_MHZ;
    }
}

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    auto type = std::make_shared<EventType>(std::move(callback), std::move(name));
    auto& registry = EventTypes();
    std::scoped_lock lock{registry.mutex};
    std::erase_if(registry.types, [](const std::weak_ptr<EventType>& entry) {
        return entry.expired();
    });
    registry.types.push_back(type);
    return type;
}

struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;
    heap_t::handle_type handle{};

    // Sort by time, unless the times are the same, in which case sort by
    // the order added to the queue
    friend bool operator>(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }

    friend bool operator<(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};

CoreTiming::CoreTiming() = default;
CoreTiming::~CoreTiming() {
    Reset();
}

void CoreTiming::Initialize(std::function<void()>&& on_thread_init_) {
    Reset();
    on_thread_init = std::move(on_thread_init_);
    event_fifo_id = 0;
    cpu_ticks = 0;
    if (is_multicore) {
        timer_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("HostTiming");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);
            Common::SetCurrentThreadToPerformanceCores();
            on_thread_init();
            has_started = true;
            thread_stats = ThreadStats{};
            thread_stats.window_start = GetGlobalTimeNs().count();
            {
                // The cheap wait is for hosts where the precise wait keeps the thread on its CPU;
                // elsewhere (sleep, condvar) the timed wait already sleeps.
                const std::string_view method = Common::EventWaitMethod();
                const bool waits_on_cpu = method == "mwaitx" || method == "umwait";
                cheap_wait_supported = waits_on_cpu && sleeper.IsAvailable();
                if (cheap_wait_supported) {
                    // Measure the timer's lateness before the first real sleep, so the margin
                    // isn't the cautious default until enough long waits come along (a few ms
                    // at boot; the margin then follows the real sleeps).
                    static constexpr s64 PROBE_NS = 200'000;
                    for (u32 probe = 0; probe < Common::SleepMargin::WARMUP; ++probe) {
                        const s64 start = GetGlobalTimeNs().count();
                        if (sleeper.SleepFor(std::chrono::nanoseconds{PROBE_NS}) !=
                            Common::TimedSleeper::Result::TimedOut) {
                            break;
                        }
                        sleep_margin.AddOvershoot(GetGlobalTimeNs().count() - start - PROBE_NS);
                    }
                }
                LOG_INFO(Core_Timing,
                         "nx2 timing wait: precise wait {}, high-resolution sleep {}: cheap wait "
                         "(A/B bit TimingWaitCheap) {}, margin {:.0f} us",
                         method, sleeper.IsAvailable() ? "available" : "unavailable",
                         cheap_wait_supported ? "used while the bit is on"
                                              : "not used (the timed wait already sleeps or no "
                                                "timer)",
                         static_cast<double>(sleep_margin.Margin()) / 1e3);
            }

            // base frequency in MHz: 1ns (10^-9) = 1GHz (10^9)
            while (!stop_token.stop_requested()) {
                while (!paused && !stop_token.stop_requested()) {
                    paused_set = false;
                    const s64 run_start = GetGlobalTimeNs().count();
                    auto const next_time = Advance();
                    s64 now = GetGlobalTimeNs().count();
                    thread_stats.run_ns += now - run_start;
                    if (next_time) {
                        // There are more events left in the queue, wait until the next event.
                        auto const wait_time = *next_time - now;
                        if (wait_time > 0) {
                            const bool woken = WaitForNextEvent(now, *next_time);
                            const s64 waited_until = GetGlobalTimeNs().count();
                            CountTimedWait(wait_time, woken, waited_until - now,
                                           waited_until - *next_time);
                            now = waited_until;
                        }
                    } else {
                        // Queue is empty, wait until another event is scheduled and signals us to
                        // continue.
                        wait_set = true;
                        event.Wait();
                        const s64 woke = GetGlobalTimeNs().count();
                        thread_stats.idle_wait_ns += woke - now;
                        ++thread_stats.idle_waits;
                        now = woke;
                    }
                    wait_set = false;
                    if (now - thread_stats.window_start >= Common::Nx2Counters::WINDOW_NS) {
                        ReportStats(now);
                    }
                }
                paused_set = true;
                const s64 pause_start = GetGlobalTimeNs().count();
                pause_event.Wait();
                thread_stats.paused_ns += GetGlobalTimeNs().count() - pause_start;
            }
        });
    }
}

void CoreTiming::ClearPendingEvents() {
    std::scoped_lock lock{advance_lock, basic_lock};
    event_queue.clear();
    event.Set();
    WakeSleepingTimer();
}

void CoreTiming::Pause(bool is_paused) {
    paused = is_paused;
    pause_event.Set();

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

void CoreTiming::SyncPause(bool is_paused) {
    if (is_paused == paused && paused_set == paused) {
        return;
    }

    Pause(is_paused);
    if (timer_thread.joinable()) {
        if (!is_paused) {
            pause_event.Set();
        }
        event.Set();
        WakeSleepingTimer();
        while (paused_set != is_paused)
            ;
    }

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

bool CoreTiming::IsRunning() const {
    return !paused_set;
}

bool CoreTiming::HasPendingEvents() const {
    std::scoped_lock lock{basic_lock};
    return !(wait_set && event_queue.empty());
}

void CoreTiming::ScheduleEvent(std::chrono::nanoseconds ns_into_future,
                               const std::shared_ptr<EventType>& event_type, bool absolute_time) {
    bool earliest{};
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? ns_into_future : GetGlobalTimeNs() + ns_into_future};

        const u64 fifo_order = event_fifo_id++;
        auto h{event_queue.emplace(Event{next_time.count(), fifo_order, event_type, 0})};
        (*h).handle = h;
        earliest = event_queue.top().fifo_order == fifo_order;
    }
    event_type->scheduled.fetch_add(1, std::memory_order_relaxed);

    WakeTimerThread(earliest);
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time,
                                      std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type,
                                      bool absolute_time) {
    bool earliest{};
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? start_time : GetGlobalTimeNs() + start_time};

        const u64 fifo_order = event_fifo_id++;
        auto h{event_queue.emplace(
            Event{next_time.count(), fifo_order, event_type, resched_time.count()})};
        (*h).handle = h;
        earliest = event_queue.top().fifo_order == fifo_order;
    }
    event_type->scheduled.fetch_add(1, std::memory_order_relaxed);

    WakeTimerThread(earliest);
}

void CoreTiming::WakeTimerThread(bool earliest) {
    // The timing thread reads the earliest event again whenever it wakes (Advance), and a wake
    // set while it isn't waiting yet ends its next wait at once. So it only needs waking for an
    // event that became the earliest; every timed guest wait schedules one, from inside the
    // scheduler lock. Off (A/B bit 57): a wake for every event, as before.
    // Idea: MythosOfs/Orchid 6e14af2079.
    if (earliest || !Common::Nx2Ab::On(Common::Nx2Ab::TimingWakeEarliest)) {
        event.Set();
        WakeSleepingTimer();
    }
}

void CoreTiming::WakeSleepingTimer() {
    // Pairs with WaitForNextEvent: it marks itself sleeping, then looks at the event; this sets
    // the event (event.Set), then looks at the mark. Both sequentially consistent, so either the
    // sleep is skipped or it is woken. A wake that finds no sleep left ends a later one early.
    if (timer_sleeping.load()) {
        sleeper.Wake();
    }
}

bool CoreTiming::WaitForNextEvent(s64 now, s64 next_time) {
    // A sleep must be worth its two system calls.
    static constexpr s64 MIN_SLEEP_NS = 100'000;
    s64 wait = next_time - now;
    if (cheap_wait_supported && Common::Nx2Ab::On(Common::Nx2Ab::TimingWaitCheap)) {
        const s64 margin = sleep_margin.Margin();
        if (wait <= margin + MIN_SLEEP_NS) {
            ++thread_stats.short_waits;
        } else {
            const s64 sleep_for = wait - margin;
            timer_sleeping.store(true);
            const bool already_set = event.IsSet();
            const auto result = already_set
                                    ? Common::TimedSleeper::Result::Woken
                                    : sleeper.SleepFor(std::chrono::nanoseconds{sleep_for});
            timer_sleeping.store(false);
            const s64 after = GetGlobalTimeNs().count();
            ThreadStats& stats = thread_stats;
            switch (result) {
            case Common::TimedSleeper::Result::Woken:
                // A new earliest event (or a pause, a stop): take the event's signal, as the
                // precise wait would have, and let Advance read the queue again.
                event.Reset();
                if (!already_set) {
                    ++stats.sleeps;
                    ++stats.sleeps_woken;
                    stats.sleep_ns += after - now;
                }
                return true;
            case Common::TimedSleeper::Result::TimedOut: {
                const s64 overshoot = after - (now + sleep_for);
                sleep_margin.AddOvershoot(overshoot);
                stats.sleep_overshoot.Add(overshoot);
                ++stats.sleeps;
                stats.sleep_ns += after - now;
                if (after >= next_time) {
                    ++stats.sleeps_past_event;
                }
                break;
            }
            case Common::TimedSleeper::Result::Failed:
                ++stats.sleep_failures;
                break;
            }
            wait = next_time - after;
            if (wait <= 0) {
                return false;
            }
        }
    }
    return event.WaitFor(std::chrono::nanoseconds{wait});
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                                 UnscheduleEventType type) {
    {
        std::scoped_lock lk{basic_lock};

        std::vector<heap_t::handle_type> to_remove;
        for (auto itr = event_queue.begin(); itr != event_queue.end(); itr++) {
            const Event& e = *itr;
            if (e.type.lock().get() == event_type.get()) {
                to_remove.push_back(itr->handle);
            }
        }

        for (auto& h : to_remove) {
            event_queue.erase(h);
        }

        event_type->sequence_number++;
        unschedule_calls.fetch_add(1, std::memory_order_relaxed);
        event_type->unscheduled.fetch_add(static_cast<u32>(to_remove.size()),
                                          std::memory_order_relaxed);
    }

    // Force any in-progress events to finish
    if (type == UnscheduleEventType::Wait) {
        std::scoped_lock lk{advance_lock};
    }
}

static u64 GetNextTickCount(u64 next_ticks) {
    if (Settings::values.use_custom_cpu_ticks.GetValue()) {
        return Settings::values.cpu_ticks.GetValue();
    }
    return next_ticks;
}

void CoreTiming::AddTicks(u64 ticks_to_add) {
    const u64 ticks = GetNextTickCount(ticks_to_add);
    cpu_ticks += ticks;
    downcount -= static_cast<s64>(ticks);
}

void CoreTiming::Idle() {
    AddTicks(1000U);
}

void CoreTiming::ResetTicks() {
    downcount = MAX_SLICE_LENGTH;
}

u64 CoreTiming::GetClockTicks() const {
    u64 fres = is_multicore ? Common::g_wall_clock.GetCNTPCT() : Common::g_wall_clock.CPUTickToGuestCNTPCT(cpu_ticks);
    if (const u32 target = CpuClockTargetMhz(Settings::values.cpu_clock.GetValue());
        target != CPU_CLOCK_BASE_MHZ) {
        fres = fres * target / CPU_CLOCK_BASE_MHZ;
    }
    if (::Settings::values.sync_core_speed.GetValue()) {
        auto const ticks = f64(fres);
        auto const speed_limit = f64(Settings::SpeedLimit()) * 0.01;
        return u64(ticks / speed_limit);
    }
    return fres;
}

u64 CoreTiming::GetGPUTicks() const {
    return is_multicore
        ? Common::g_wall_clock.GetGPUTick()
        : Common::WallClock::CPUTickToGPUTick(cpu_ticks);
}

std::optional<s64> CoreTiming::Advance() {
    std::scoped_lock lock{advance_lock, basic_lock};
    global_timer = GetGlobalTimeNs().count();

    while (!event_queue.empty() && event_queue.top().time <= global_timer) {
        const Event& evt = event_queue.top();

        if (const auto event_type{evt.type.lock()}) {
            const auto evt_time = evt.time;
            const auto evt_sequence_num = event_type->sequence_number;
            event_type->fired.fetch_add(1, std::memory_order_relaxed);

            if (evt.reschedule_time == 0) {
                event_queue.pop();

                basic_lock.unlock();

                const s64 late = GetGlobalTimeNs().count() - evt_time;
                CountLate(late);
                event_type->callback(evt_time, std::chrono::nanoseconds{late});

                basic_lock.lock();
            } else {
                basic_lock.unlock();

                const s64 late = GetGlobalTimeNs().count() - evt_time;
                CountLate(late);
                const auto new_schedule_time{
                    event_type->callback(evt_time, std::chrono::nanoseconds{late})};

                basic_lock.lock();

                if (evt_sequence_num != event_type->sequence_number) {
                    // Heap handle is invalidated after external modification.
                    continue;
                }

                const auto next_schedule_time{new_schedule_time.has_value()
                                                  ? new_schedule_time.value().count()
                                                  : evt.reschedule_time};

                // If this event was scheduled into a pause, its time now is going to be way
                // behind. Re-set this event to continue from the end of the pause.
                auto next_time{evt.time + next_schedule_time};
                if (evt.time < pause_end_time) {
                    next_time = pause_end_time + next_schedule_time;
                }

                event_queue.update(evt.handle, Event{next_time, event_fifo_id++, evt.type,
                                                     next_schedule_time, evt.handle});
            }
        }

        global_timer = GetGlobalTimeNs().count();
    }

    if (!event_queue.empty()) {
        return event_queue.top().time;
    } else {
        return std::nullopt;
    }
}

void CoreTiming::CountLate(s64 late) {
    ++late_count;
    late_ns += late;
    late_max_ns = std::max(late_max_ns, late);
    late_histogram.Add(late);
}

void CoreTiming::CountTimedWait(s64 wait_ns, bool woken, s64 waited_ns, s64 overshoot_ns) {
    ThreadStats& stats = thread_stats;
    ++stats.timed_waits;
    stats.timed_wait_ns += waited_ns;
    if (wait_ns < 50'000) {
        ++stats.waits_under_50us;
    } else if (wait_ns < 500'000) {
        ++stats.waits_under_500us;
    }
    if (woken) {
        ++stats.woken_waits;
    } else if (overshoot_ns > 0) {
        ++stats.overshoots;
        stats.overshoot_ns += overshoot_ns;
    }
}

void CoreTiming::ReportStats(s64 now) {
    using namespace Common::Nx2Counters;
    const ThreadStats stats = thread_stats;
    const s64 active = std::max<s64>(now - stats.window_start - stats.paused_ns, 1);

    // Per event type, summed by name (some names have several types).
    struct Row {
        std::string name;
        u64 scheduled;
        u64 fired;
        u64 unscheduled;
    };
    std::vector<std::weak_ptr<EventType>> types;
    {
        auto& registry = EventTypes();
        std::scoped_lock lock{registry.mutex};
        types = registry.types;
    }
    std::vector<Row> rows;
    {
        for (const std::weak_ptr<EventType>& entry : types) {
            const std::shared_ptr<EventType> type = entry.lock();
            if (!type) {
                continue;
            }
            const u64 scheduled = type->scheduled.exchange(0, std::memory_order_relaxed);
            const u64 fired = type->fired.exchange(0, std::memory_order_relaxed);
            const u64 unscheduled = type->unscheduled.exchange(0, std::memory_order_relaxed);
            if (scheduled + fired + unscheduled == 0) {
                continue;
            }
            const auto it = std::find_if(rows.begin(), rows.end(),
                                         [&](const Row& row) { return row.name == type->name; });
            if (it == rows.end()) {
                rows.push_back(Row{type->name, scheduled, fired, unscheduled});
            } else {
                it->scheduled += scheduled;
                it->fired += fired;
                it->unscheduled += unscheduled;
            }
        }
    }
    Row total{"", 0, 0, 0};
    for (const Row& row : rows) {
        total.scheduled += row.scheduled;
        total.fired += row.fired;
        total.unscheduled += row.unscheduled;
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.scheduled + a.fired > b.scheduled + b.fired;
    });
    std::string top;
    for (size_t i = 0; i < std::min<size_t>(rows.size(), 10); ++i) {
        top += fmt::format("{}{} {}/{}/{}", i == 0 ? "" : ", ", rows[i].name,
                           FormatRate(PerSecond(rows[i].scheduled, active)),
                           FormatRate(PerSecond(rows[i].fired, active)),
                           FormatRate(PerSecond(rows[i].unscheduled, active)));
    }
    size_t queued{};
    {
        std::scoped_lock lock{basic_lock};
        queued = event_queue.size();
    }

    // The timed waits keep the thread on its CPU with MWAITX/UMWAIT (Windows on x86), so "on the
    // CPU" counts them; with "sleep" or "condvar" the thread sleeps in them.
    const std::string_view method = Common::EventWaitMethod();
    const bool waits_on_cpu = method == "mwaitx" || method == "umwait";
    const u64 waits = stats.timed_waits + stats.idle_waits;
    const auto share = [&](u64 count) {
        return stats.timed_waits != 0 ? static_cast<double>(count) * 100.0 /
                                            static_cast<double>(stats.timed_waits)
                                      : 0.0;
    };
    LOG_INFO(Core_Timing,
             "nx2 timing (last {:.1f} s, paused {:.1f} s): events scheduled {}/s, fired {}/s, "
             "unscheduled {}/s ({} unschedule calls/s), {} queued, fired late p50 {:.1f} us "
             "p99 {:.1f} us mean {:.1f} us max {:.1f} us | HostTiming run {:.1f}%, timed wait "
             "{:.1f}% ({}, slept {:.1f}%), idle wait {:.1f}%, on the CPU {:.1f}% | waits {}/s: "
             "timed {}/s (<50 us {:.0f}%, 50-500 us "
             "{:.0f}%, >=500 us {:.0f}%, ended by a new event {:.0f}%, ran out {:.1f} us late "
             "on average), idle {}/s | top 10 by scheduled+fired (scheduled/fired/unscheduled "
             "per s): {}",
             static_cast<double>(active) / 1e9, static_cast<double>(stats.paused_ns) / 1e9,
             FormatRate(PerSecond(total.scheduled, active)),
             FormatRate(PerSecond(total.fired, active)),
             FormatRate(PerSecond(total.unscheduled, active)),
             FormatRate(PerSecond(unschedule_calls.exchange(0, std::memory_order_relaxed), active)),
             queued, static_cast<double>(late_histogram.Percentile(0.5)) / 1e3,
             static_cast<double>(late_histogram.Percentile(0.99)) / 1e3,
             late_count != 0 ? static_cast<double>(late_ns) / static_cast<double>(late_count) / 1e3
                             : 0.0,
             static_cast<double>(late_max_ns) / 1e3, Percent(stats.run_ns, active),
             Percent(stats.timed_wait_ns, active), method, Percent(stats.sleep_ns, active),
             Percent(stats.idle_wait_ns, active),
             Percent(stats.run_ns + (waits_on_cpu ? stats.timed_wait_ns - stats.sleep_ns : 0),
                     active),
             FormatRate(PerSecond(waits, active)),
             FormatRate(PerSecond(stats.timed_waits, active)), share(stats.waits_under_50us),
             share(stats.waits_under_500us),
             share(stats.timed_waits - stats.waits_under_50us - stats.waits_under_500us),
             share(stats.woken_waits),
             stats.overshoots != 0 ? static_cast<double>(stats.overshoot_ns) /
                                         static_cast<double>(stats.overshoots) / 1e3
                                   : 0.0,
             FormatRate(PerSecond(stats.idle_waits, active)), top.empty() ? "none" : top);

    // The cheap wait (A/B bit TimingWaitCheap): how often it slept, and how close to the events
    // the sleeps ended. "past the event" = the sleep alone overran the event's time (it fired late).
    if (cheap_wait_supported) {
        const auto per_timed = [&](u64 count) {
            return stats.timed_waits != 0 ? static_cast<double>(count) * 100.0 /
                                                static_cast<double>(stats.timed_waits)
                                          : 0.0;
        };
        LOG_INFO(Core_Timing,
                 "nx2 timing wait (A/B bit TimingWaitCheap {}): slept {:.1f}% of the time in "
                 "{}/s sleeps ({:.0f}% of timed waits; {:.0f}% ended by a new event), timer late "
                 "p50 {:.1f} us p99 {:.1f} us, margin now {:.0f} us, {} sleeps past their event, "
                 "{}/s waits too short to sleep ({:.0f}%), {} timer failures",
                 Common::Nx2Ab::On(Common::Nx2Ab::TimingWaitCheap) ? "on" : "off",
                 Percent(stats.sleep_ns, active), FormatRate(PerSecond(stats.sleeps, active)),
                 per_timed(stats.sleeps),
                 stats.sleeps != 0 ? static_cast<double>(stats.sleeps_woken) * 100.0 /
                                         static_cast<double>(stats.sleeps)
                                   : 0.0,
                 static_cast<double>(stats.sleep_overshoot.Percentile(0.5)) / 1e3,
                 static_cast<double>(stats.sleep_overshoot.Percentile(0.99)) / 1e3,
                 static_cast<double>(sleep_margin.Margin()) / 1e3, stats.sleeps_past_event,
                 FormatRate(PerSecond(stats.short_waits, active)), per_timed(stats.short_waits),
                 stats.sleep_failures);
    }

    // The guest cores' SVC and hint counts, on the same 30 s.
    Core::Nx2CpuStats::Report(active);

    thread_stats = ThreadStats{};
    thread_stats.window_start = now;
    late_count = 0;
    late_ns = 0;
    late_max_ns = 0;
    late_histogram.Clear();
}

void CoreTiming::Reset() {
    paused = true;
    // Request the stop before waking the thread: woken first, it could take the wake-up, see no
    // stop request yet, and go back to waiting on pause_event for good (the join below then hung
    // the frontend at boot).
    if (timer_thread.joinable()) {
        timer_thread.request_stop();
    }
    pause_event.Set();
    event.Set();
    sleeper.Wake();
    if (timer_thread.joinable()) {
        timer_thread.join();
    }
    has_started = false;
}

/// @brief Returns current time in nanoseconds.
std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeNS()
        : std::chrono::nanoseconds{Common::WallClock::CPUTickToNS(cpu_ticks)};
}

/// @brief Returns current time in microseconds.
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeUS()
        : std::chrono::microseconds{Common::WallClock::CPUTickToUS(cpu_ticks)};
}

#ifdef _WIN32
void CoreTiming::SetTimerResolutionNs(std::chrono::nanoseconds ns) {
    timer_resolution_ns = ns.count();
}
#endif

} // namespace Core::Timing
