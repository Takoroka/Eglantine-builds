// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <QDialog>
#include <QString>

#include "common/common_types.h"
#include "qt_common/util/mod_list.h"

class QLabel;
class QLineEdit;
class QPushButton;
class QTreeWidget;

// The community mod list for the games found in the game list: install, update and remove mods
// into load/<title id>/, where the Add-Ons tab manages them like any other mod.
class ModListDialog : public QDialog {
    Q_OBJECT

public:
    struct Game {
        u64 title_id;
        QString name;
    };

    explicit ModListDialog(std::vector<Game> games_, QWidget* parent = nullptr);
    ~ModListDialog() override;

    // Whether a mod was installed or removed (the game list's add-ons need a refresh).
    [[nodiscard]] bool Changed() const {
        return changed;
    }

private:
    void Populate();
    void UpdateDetails();
    void OnRefresh();
    void OnInstall();
    void OnRemove();
    void OpenModsFolder();
    void Finish(const QString& error, const QString& done, u64 title_id);

    // The selected mod and the game row it's under; nullptr when a game row or nothing is selected.
    const QtCommon::ModList::Mod* SelectedMod(u64* title_id = nullptr) const;

    std::vector<Game> games;
    QtCommon::ModList::Index index;
    bool changed = false;

    QTreeWidget* tree;
    QLabel* empty_label;
    QLabel* title_label;
    QLabel* info_label;
    QLabel* description_label;
    QLabel* status_label;
    QPushButton* install_button;
    QPushButton* remove_button;
    QLineEdit* repo_url;
};
