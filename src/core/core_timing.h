// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <boost/heap/fibonacci_heap.hpp>

#include "common/common_types.h"
#include "common/nx2_counters.h"
#include "common/thread.h"
#include "common/timed_sleeper.h"
#include "common/cpu_features.h"

namespace Core::Timing {

/// A callback that may be scheduled for a particular core timing event.
using TimedCallback = std::function<std::optional<std::chrono::nanoseconds>(
    s64 time, std::chrono::nanoseconds ns_late)>;

/// Contains the characteristics of a particular event.
struct EventType {
    explicit EventType(TimedCallback&& callback_, std::string&& name_)
        : callback{std::move(callback_)}, name{std::move(name_)}, sequence_number{0} {}

    /// The event's callback function.
    TimedCallback callback;
    /// A pointer to the name of the event.
    const std::string name;
    /// A monotonic sequence number, incremented when this event is
    /// changed externally.
    size_t sequence_number;

    // TEMP (measurement aid): counts for the "nx2 timing" log line, relaxed; the timing thread
    // takes and resets them every 30 s.
    std::atomic<u32> scheduled{0};   ///< ScheduleEvent / ScheduleLoopingEvent calls
    std::atomic<u32> fired{0};       ///< Callbacks run
    std::atomic<u32> unscheduled{0}; ///< Queued events UnscheduleEvent removed
};

enum class UnscheduleEventType {
    Wait,
    NoWait,
};

/**
 * This is a system to schedule events into the emulated machine's future. Time is measured
 * in main CPU clock cycles.
 *
 * To schedule an event, you first have to register its type. This is where you pass in the
 * callback. You then schedule events using the type ID you get back.
 *
 * The s64 ns_late that the callbacks get is how many ns late it was.
 * So to schedule a new event on a regular basis:
 * inside callback:
 *   ScheduleEvent(period_in_ns - ns_late, callback, "whatever")
 */
class CoreTiming {
public:
    CoreTiming();
    ~CoreTiming();

    CoreTiming(const CoreTiming&) = delete;
    CoreTiming(CoreTiming&&) = delete;

    CoreTiming& operator=(const CoreTiming&) = delete;
    CoreTiming& operator=(CoreTiming&&) = delete;

    /// CoreTiming begins at the boundary of timing slice -1. An initial call to Advance() is
    /// required to end slice - 1 and start slice 0 before the first cycle of code is executed.
    void Initialize(std::function<void()>&& on_thread_init_);

    /// Clear all pending events. This should ONLY be done on exit.
    void ClearPendingEvents();

    /// Sets if emulation is multicore or single core, must be set before Initialize
    void SetMulticore(bool is_multicore_) {
        is_multicore = is_multicore_;
    }

    /// Pauses/Unpauses the execution of the timer thread.
    void Pause(bool is_paused);

    /// Pauses/Unpauses the execution of the timer thread and waits until paused.
    void SyncPause(bool is_paused);

    /// Checks if core timing is running.
    bool IsRunning() const;

    /// Checks if the timer thread has started.
    bool HasStarted() const {
        return has_started;
    }

    /// Checks if there are any pending time events.
    bool HasPendingEvents() const;

    /// Schedules an event in core timing
    void ScheduleEvent(std::chrono::nanoseconds ns_into_future,
                       const std::shared_ptr<EventType>& event_type, bool absolute_time = false);

    /// Schedules an event which will automatically re-schedule itself with the given time, until
    /// unscheduled
    void ScheduleLoopingEvent(std::chrono::nanoseconds start_time,
                              std::chrono::nanoseconds resched_time,
                              const std::shared_ptr<EventType>& event_type,
                              bool absolute_time = false);

    void UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                         UnscheduleEventType type = UnscheduleEventType::Wait);

    void AddTicks(u64 ticks_to_add);

    void ResetTicks();

    void Idle();

    s64 GetDowncount() const noexcept {
        return downcount;
    }

    /// Returns the current CNTPCT tick value.
    u64 GetClockTicks() const;

    /// Returns the current GPU tick value.
    u64 GetGPUTicks() const;

    [[nodiscard]] std::chrono::microseconds GetGlobalTimeUs() const noexcept;
    [[nodiscard]] std::chrono::nanoseconds GetGlobalTimeNs() const noexcept;

    /// Checks for events manually and returns time in nanoseconds for next event, threadsafe.
    std::optional<s64> Advance();

#ifdef _WIN32
    void SetTimerResolutionNs(std::chrono::nanoseconds ns);
#endif

    struct Event;

    void Reset();
    /// Wakes the timing thread after an event was scheduled; earliest: it became the next one.
    void WakeTimerThread(bool earliest);

    /// TEMP (measurement aid): where the timing thread's time goes, for the "nx2 timing" line
    /// (that thread only; ReportStats logs and resets it every 30 s).
    struct ThreadStats {
        s64 window_start{};
        s64 run_ns{};        ///< In Advance: the queue and the callbacks
        s64 timed_wait_ns{}; ///< In event.WaitFor until the next event
        s64 idle_wait_ns{};  ///< In event.Wait with nothing queued
        s64 paused_ns{};     ///< Paused (left out of the rates)
        u64 timed_waits{};
        u64 woken_waits{}; ///< Timed waits a new earliest event ended (event.Set)
        u64 waits_under_50us{};
        u64 waits_under_500us{}; ///< 50-500 us: shorter than a Windows timer tick
        u64 idle_waits{};
        u64 overshoots{};   ///< Timed waits that ran out
        s64 overshoot_ns{}; ///< How long after the event's time those returned
        // The cheap wait (A/B bit TimingWaitCheap):
        u64 sleeps{};             ///< Timed waits that slept first
        s64 sleep_ns{};           ///< Time in those sleeps
        u64 sleeps_woken{};       ///< Sleeps a new earliest event ended
        u64 sleeps_past_event{};  ///< Sleeps that returned after the event's time (it ran late)
        u64 sleep_failures{};     ///< The timer could not be armed: precise wait only
        u64 short_waits{};        ///< Timed waits shorter than margin + MIN_SLEEP: precise only
        Common::Nx2Counters::LatencyHistogram sleep_overshoot{}; ///< Timer lateness per sleep
    };
    void CountTimedWait(s64 wait_ns, bool woken, s64 waited_ns, s64 overshoot_ns);
    void CountLate(s64 late);
    void ReportStats(s64 now);
    ThreadStats thread_stats{};
    /// Lateness of the callbacks Advance ran (the timing thread's when multicore).
    u64 late_count{};
    s64 late_ns{};
    s64 late_max_ns{};
    Common::Nx2Counters::LatencyHistogram late_histogram{};
    std::atomic<u32> unschedule_calls{0}; ///< UnscheduleEvent calls (each walks the queue)

    /// Waits for the next event at next_time (now: the time already read); true when a new
    /// event (event.Set) ended the wait. With A/B bit TimingWaitCheap on, where the precise wait
    /// keeps the thread on its CPU (MWAITX/UMWAIT), it first sleeps on a high-resolution timer
    /// until sleep_margin before the event, then waits precisely for the rest.
    bool WaitForNextEvent(s64 now, s64 next_time);
    /// Ends a sleep of the timing thread after event.Set (only when it sleeps).
    void WakeSleepingTimer();
    Common::TimedSleeper sleeper;
    Common::SleepMargin sleep_margin; ///< Timing thread only
    std::atomic<bool> timer_sleeping{false};
    bool cheap_wait_supported{}; ///< Set by the timing thread when it starts

    using heap_t = boost::heap::fibonacci_heap<CoreTiming::Event, boost::heap::compare<std::greater<>>>;
    heap_t event_queue;
    s64 global_timer = 0;
#ifdef _WIN32
    s64 timer_resolution_ns;
#endif
    u64 event_fifo_id = 0;
    s64 pause_end_time{};
    /// Cycle timing
    u64 cpu_ticks{};
    s64 downcount{};
    Common::Event event{};
    Common::Event pause_event{};
    std::function<void()> on_thread_init{};
    std::jthread timer_thread;
    mutable std::mutex basic_lock;
    std::mutex advance_lock;
    std::atomic<bool> paused{};
    std::atomic<bool> paused_set{};
    std::atomic<bool> wait_set{};
    std::atomic<bool> has_started{};
    bool is_multicore{};
};

/// Creates a core timing event with the given name and callback.
///
/// @param name     The name of the core timing event to create.
/// @param callback The callback to execute for the event.
///
/// @returns An EventType instance representing the created event.
///
std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback);

} // namespace Core::Timing
