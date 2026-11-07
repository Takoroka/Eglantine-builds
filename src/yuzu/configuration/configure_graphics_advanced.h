// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <vector>
#include <QWidget>
#include "yuzu/configuration/configuration_shared.h"

class QLabel;
class QSpinBox;

namespace Core {
class System;
}

namespace VkDeviceInfo {
class Record;
}

namespace Ui {
class ConfigureGraphicsAdvanced;
}

namespace ConfigurationShared {
class Builder;
class Widget;
}

class ConfigureGraphicsAdvanced : public ConfigurationShared::Tab {
    Q_OBJECT

public:
    explicit ConfigureGraphicsAdvanced(
        const Core::System& system_, const std::vector<VkDeviceInfo::Record>& records,
        std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group,
        const ConfigurationShared::Builder& builder, QWidget* parent = nullptr);
    ~ConfigureGraphicsAdvanced() override;

    void ApplyConfiguration() override;
    void SetConfiguration() override;

    void ExposeComputeOption();

private:
    void Setup(const ConfigurationShared::Builder& builder);
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    /// Wraps the Texture Cache Memory Limit widget with a note: what Auto is on the selected
    /// Vulkan device, or why a value is raised, lowered or below the recommendation.
    QWidget* WrapMemoryLimit(ConfigurationShared::Widget* widget);
    void UpdateMemoryLimitNote();

    std::unique_ptr<Ui::ConfigureGraphicsAdvanced> ui;

    const Core::System& system;
    const std::vector<VkDeviceInfo::Record>& records;

    QSpinBox* memory_limit_spinbox{};
    QLabel* memory_limit_note{};

    std::vector<std::function<void(bool)>> apply_funcs;

    QWidget* checkbox_enable_compute_pipelines{};
};
