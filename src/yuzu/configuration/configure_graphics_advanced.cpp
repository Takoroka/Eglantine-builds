// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>
#include <QLabel>
#include <QSpinBox>
#include <QVBoxLayout>
#include <qnamespace.h>
#include "common/literals.h"
#include "common/settings.h"
#include "core/core.h"
#include "qt_common/config/shared_translation.h"
#include "qt_common/util/vk.h"
#include "ui_configure_graphics_advanced.h"
#include "video_core/texture_cache/memory_limit.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_graphics_advanced.h"
#include "yuzu/configuration/shared_widget.h"

ConfigureGraphicsAdvanced::ConfigureGraphicsAdvanced(
    const Core::System& system_, const std::vector<VkDeviceInfo::Record>& records_,
    std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group_, parent), ui{std::make_unique<Ui::ConfigureGraphicsAdvanced>()}, system{system_},
      records{records_} {

    ui->setupUi(this);

    Setup(builder);

    SetConfiguration();

    checkbox_enable_compute_pipelines->setVisible(false);
}

ConfigureGraphicsAdvanced::~ConfigureGraphicsAdvanced() = default;

void ConfigureGraphicsAdvanced::SetConfiguration() {}

void ConfigureGraphicsAdvanced::Setup(const ConfigurationShared::Builder& builder) {
    auto& normal_layout = *ui->normal_target->layout();

    // A map will sort the data for us
    std::map<u32, QWidget*> normal_hold{};

    for (auto setting :
         Settings::values.linkage.by_category[Settings::Category::RendererAdvanced]) {
        const bool is_memory_limit =
            setting->Id() == Settings::values.texture_cache_memory_limit.Id();
        ConfigurationShared::Widget* widget =
            is_memory_limit
                ? builder.BuildWidget(setting, apply_funcs, ConfigurationShared::RequestType::SpinBox,
                                      true, 1.0f, nullptr, tr(" MiB"))
                : builder.BuildWidget(setting, apply_funcs);

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        const auto id = setting->Id();

        normal_hold.emplace(id, is_memory_limit ? WrapMemoryLimit(widget) : widget);

        // Keep track of enable_compute_pipelines so we can display it when needed
        if (setting->Id() == Settings::values.enable_compute_pipelines.Id()) {
            checkbox_enable_compute_pipelines = widget;
        }
    }

    for (const auto& [id, widget] : normal_hold) {
        normal_layout.addWidget(widget);
    }
}

QWidget* ConfigureGraphicsAdvanced::WrapMemoryLimit(ConfigurationShared::Widget* widget) {
    memory_limit_spinbox = widget->spinbox;
    if (memory_limit_spinbox == nullptr) {
        return widget;
    }
    // 0 is Auto; the spinbox stops at the selected GPU's memory.
    memory_limit_spinbox->setSpecialValueText(tr("Auto"));
    memory_limit_spinbox->setSingleStep(256);
    const u32 device = Settings::values.vulkan_device.GetValue();
    if (device < records.size() && records[device].device_memory != 0) {
        memory_limit_spinbox->setMaximum(static_cast<int>(records[device].device_memory >> 20));
    }

    auto* container = new QWidget(this);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    layout->addWidget(widget);
    memory_limit_note = new QLabel(container);
    memory_limit_note->setWordWrap(true);
    layout->addWidget(memory_limit_note);

    connect(memory_limit_spinbox, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int) { UpdateMemoryLimitNote(); });
    UpdateMemoryLimitNote();
    return container;
}

void ConfigureGraphicsAdvanced::UpdateMemoryLimitNote() {
    using namespace Common::Literals;
    if (memory_limit_spinbox == nullptr || memory_limit_note == nullptr) {
        return;
    }
    const u32 value = static_cast<u32>(memory_limit_spinbox->value());
    const u32 device = Settings::values.vulkan_device.GetValue();
    const VkDeviceInfo::Record* record = device < records.size() ? &records[device] : nullptr;
    if (record == nullptr || record->device_memory == 0) {
        constexpr u32 min_mib = static_cast<u32>(VideoCommon::MinTextureCacheMemoryLimit >> 20);
        memory_limit_note->setStyleSheet(QString());
        memory_limit_note->setText(
            value == 0        ? tr("Auto: three quarters of the GPU's memory budget.")
            : value < min_mib ? tr("Raised to %1 MiB, the smallest limit.").arg(min_mib)
                              : QString());
        return;
    }
    // The texture cache follows the budget of an integrated GPU only up to its share of the RAM
    // (Vulkan::Device::GetDeviceMemoryBudget).
    u64 budget = record->memory_budget;
    if (record->integrated) {
        const bool aggressive =
            Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive;
        budget = (std::min)(budget, aggressive ? u64{6_GiB} : u64{4_GiB});
    }
    const auto limit =
        VideoCommon::ResolveTextureCacheMemoryLimit(value, budget, record->device_memory);
    QString text;
    if (limit.automatic) {
        text = tr("Auto: three quarters of this GPU's %1 MiB budget (%2 MiB now, never less than "
                  "%3 MiB), following it in game. Switch 2 games on a dedicated GPU collect near "
                  "the budget itself.")
                   .arg(budget >> 20)
                   .arg(limit.limit >> 20)
                   .arg(VideoCommon::FixedTextureCacheCriticalCap >> 20);
    } else if (limit.raised) {
        text = tr("Raised to %1 MiB, the smallest limit.").arg(limit.limit >> 20);
    } else if (limit.clamped) {
        text = tr("Lowered to %1 MiB, this GPU's memory.").arg(limit.limit >> 20);
    }
    const bool warn = !limit.automatic && limit.BelowRecommended();
    if (warn) {
        if (!text.isEmpty()) {
            text += QLatin1Char(' ');
        }
        text += tr("Below the recommended %1 MiB: textures are evicted and decoded again sooner, "
                   "which can stutter.")
                    .arg(limit.recommended >> 20);
    }
    memory_limit_note->setStyleSheet(warn ? QStringLiteral("color: #d35400;") : QString());
    memory_limit_note->setText(text);
}

void ConfigureGraphicsAdvanced::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(is_powered_on);
    }
}

void ConfigureGraphicsAdvanced::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureGraphicsAdvanced::RetranslateUI() {
    ui->retranslateUi(this);
    UpdateMemoryLimitNote();
}

void ConfigureGraphicsAdvanced::ExposeComputeOption() {
    checkbox_enable_compute_pipelines->setVisible(true);
}
