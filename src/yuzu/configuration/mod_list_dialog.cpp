// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <filesystem>
#include <system_error>

#include <QApplication>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSplitter>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <fmt/format.h>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/settings.h"
#include "yuzu/configuration/mod_list_dialog.h"

namespace {
constexpr int ModRole = Qt::UserRole;
constexpr int TitleRole = Qt::UserRole + 1;

QString TitleIdText(u64 title_id) {
    return QString::fromStdString(fmt::format("{:016X}", title_id));
}

QString StateText(QtCommon::ModList::State state) {
    switch (state) {
    case QtCommon::ModList::State::NotInstalled:
        return QObject::tr("Not installed");
    case QtCommon::ModList::State::Installed:
        return QObject::tr("Installed");
    case QtCommon::ModList::State::UpdateAvailable:
        return QObject::tr("Update available");
    case QtCommon::ModList::State::Foreign:
        return QObject::tr("Folder exists");
    }
    return {};
}
} // Anonymous namespace

ModListDialog::ModListDialog(std::vector<Game> games_, QWidget* parent)
    : QDialog(parent), games{std::move(games_)} {
    setWindowTitle(tr("Mod List"));
    resize(860, 540);

    tree = new QTreeWidget(this);
    tree->setColumnCount(4);
    tree->setHeaderLabels({tr("Mod"), tr("Version"), tr("Author"), tr("Status")});
    tree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    tree->header()->setStretchLastSection(false);

    empty_label = new QLabel(this);
    empty_label->setWordWrap(true);

    title_label = new QLabel(this);
    QFont title_font = title_label->font();
    title_font.setBold(true);
    title_label->setFont(title_font);
    title_label->setWordWrap(true);
    info_label = new QLabel(this);
    info_label->setWordWrap(true);
    description_label = new QLabel(this);
    description_label->setWordWrap(true);
    description_label->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    install_button = new QPushButton(tr("Install"), this);
    remove_button = new QPushButton(tr("Remove"), this);

    auto* details = new QWidget(this);
    auto* details_layout = new QVBoxLayout(details);
    details_layout->addWidget(title_label);
    details_layout->addWidget(info_label);
    details_layout->addWidget(description_label, 1);
    auto* mod_buttons = new QHBoxLayout;
    mod_buttons->addWidget(install_button);
    mod_buttons->addWidget(remove_button);
    details_layout->addLayout(mod_buttons);

    auto* list = new QWidget(this);
    auto* list_layout = new QVBoxLayout(list);
    list_layout->setContentsMargins(0, 0, 0, 0);
    list_layout->addWidget(tree, 1);
    list_layout->addWidget(empty_label);

    auto* splitter = new QSplitter(this);
    splitter->addWidget(list);
    splitter->addWidget(details);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);

    status_label = new QLabel(this);
    status_label->setWordWrap(true);

    repo_url = new QLineEdit(QString::fromStdString(Settings::values.mod_list_repo_url.GetValue()),
                             this);
    auto* refresh = new QPushButton(tr("Refresh list"), this);
    auto* repo_row = new QHBoxLayout;
    repo_row->addWidget(new QLabel(tr("Mod repository:"), this));
    repo_row->addWidget(repo_url, 1);
    repo_row->addWidget(refresh);

    auto* open_folder = new QPushButton(tr("Open mods folder"), this);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    auto* bottom_row = new QHBoxLayout;
    bottom_row->addWidget(open_folder);
    bottom_row->addStretch(1);
    bottom_row->addWidget(buttons);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(splitter, 1);
    layout->addWidget(status_label);
    layout->addLayout(repo_row);
    layout->addLayout(bottom_row);

    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(refresh, &QPushButton::clicked, this, &ModListDialog::OnRefresh);
    connect(install_button, &QPushButton::clicked, this, &ModListDialog::OnInstall);
    connect(remove_button, &QPushButton::clicked, this, &ModListDialog::OnRemove);
    connect(open_folder, &QPushButton::clicked, this, &ModListDialog::OpenModsFolder);
    connect(tree, &QTreeWidget::itemSelectionChanged, this, &ModListDialog::UpdateDetails);
    connect(repo_url, &QLineEdit::editingFinished, this, [this] {
        Settings::values.mod_list_repo_url.SetValue(repo_url->text().trimmed().toStdString());
    });

    // Only what an earlier refresh downloaded: nothing is fetched until Refresh list.
    index = QtCommon::ModList::LoadCached();
    Populate();
}

ModListDialog::~ModListDialog() = default;

void ModListDialog::Populate() {
    tree->clear();
    std::size_t shown = 0;
    for (const auto& game : games) {
        QTreeWidgetItem* game_item = nullptr;
        for (std::size_t i = 0; i < index.mods.size(); ++i) {
            const auto& mod = index.mods[i];
            if (std::find(mod.title_ids.begin(), mod.title_ids.end(), game.title_id) ==
                mod.title_ids.end()) {
                continue;
            }
            if (game_item == nullptr) {
                const auto name = game.name.isEmpty() ? TitleIdText(game.title_id)
                                                      : QStringLiteral("%1 (%2)").arg(
                                                            game.name, TitleIdText(game.title_id));
                game_item = new QTreeWidgetItem(tree, {name});
            }
            auto* item = new QTreeWidgetItem(
                game_item, {mod.name, mod.version, mod.author,
                            StateText(QtCommon::ModList::GetState(mod, game.title_id))});
            item->setData(0, ModRole, static_cast<qulonglong>(i));
            item->setData(0, TitleRole, static_cast<qulonglong>(game.title_id));
            ++shown;
        }
    }
    tree->expandAll();
    tree->setVisible(shown != 0);
    empty_label->setVisible(shown == 0);
    if (index.mods.empty()) {
        empty_label->setText(
            tr("Click Refresh list to download the community mod list from the mod repository."));
    } else {
        empty_label->setText(tr("The mod list has no mods for the games in your game list."));
    }
    UpdateDetails();
}

const QtCommon::ModList::Mod* ModListDialog::SelectedMod(u64* title_id) const {
    const auto selected = tree->selectedItems();
    if (selected.isEmpty() || !selected.front()->data(0, ModRole).isValid()) {
        return nullptr;
    }
    if (title_id != nullptr) {
        *title_id = selected.front()->data(0, TitleRole).toULongLong();
    }
    return &index.mods[selected.front()->data(0, ModRole).toULongLong()];
}

void ModListDialog::UpdateDetails() {
    u64 title_id = 0;
    const auto* mod = SelectedMod(&title_id);
    install_button->setEnabled(false);
    remove_button->setEnabled(false);
    if (mod == nullptr) {
        title_label->setText(tree->isVisible() ? tr("Select a mod.") : QString{});
        info_label->clear();
        description_label->clear();
        return;
    }
    const auto state = QtCommon::ModList::GetState(*mod, title_id);
    title_label->setText(mod->name);
    QStringList info;
    if (!mod->category.isEmpty()) {
        info.push_back(tr("Category: %1").arg(mod->category));
    }
    info.push_back(tr("Version: %1").arg(mod->version));
    info.push_back(tr("Author: %1").arg(mod->author));
    info.push_back(tr("Status: %1").arg(StateText(state)));
    info.push_back(tr("Installs to: load/%1/%2").arg(TitleIdText(title_id), mod->FolderName()));
    info_label->setText(info.join(QLatin1Char('\n')));
    description_label->setText(mod->description);

    install_button->setText(state == QtCommon::ModList::State::UpdateAvailable ? tr("Update")
                                                                               : tr("Install"));
    install_button->setEnabled(state == QtCommon::ModList::State::NotInstalled ||
                               state == QtCommon::ModList::State::UpdateAvailable);
    remove_button->setEnabled(state == QtCommon::ModList::State::Installed ||
                              state == QtCommon::ModList::State::UpdateAvailable);
}

void ModListDialog::OnRefresh() {
    const auto url = repo_url->text().trimmed();
    Settings::values.mod_list_repo_url.SetValue(url.toStdString());
    status_label->setText(tr("Downloading the mod list..."));
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();
    auto fresh = QtCommon::ModList::Refresh(url.toStdString());
    QApplication::restoreOverrideCursor();
    if (!fresh.error.isEmpty()) {
        status_label->setText(fresh.error);
        QMessageBox::warning(this, tr("Mod List"), fresh.error);
        return;
    }
    index = std::move(fresh);
    status_label->setText(tr("The mod list has %n mod(s).", "", static_cast<int>(index.mods.size())));
    Populate();
}

void ModListDialog::Finish(const QString& error, const QString& done, u64 title_id) {
    if (!error.isEmpty()) {
        status_label->setText(error);
        QMessageBox::warning(this, tr("Mod List"), error);
    } else {
        status_label->setText(done);
        changed = true;
        // The game list caches each game's add-ons.
        Common::FS::RemoveFile(Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) /
                               "game_list" / fmt::format("{:016X}.pv.txt", title_id));
    }
    Populate();
}

void ModListDialog::OnInstall() {
    u64 title_id = 0;
    const auto* mod = SelectedMod(&title_id);
    if (mod == nullptr) {
        return;
    }
    const auto name = mod->name;
    const auto error = QtCommon::ModList::Install(*mod, title_id);
    Finish(error,
           tr("Installed %1. It's enabled in the game's Add-Ons tab and applies when the game "
              "boots.")
               .arg(name),
           title_id);
}

void ModListDialog::OnRemove() {
    u64 title_id = 0;
    const auto* mod = SelectedMod(&title_id);
    if (mod == nullptr) {
        return;
    }
    const auto name = mod->name;
    if (QMessageBox::question(this, tr("Remove mod"),
                              tr("Remove %1 from load/%2? Its folder is deleted.")
                                  .arg(name, TitleIdText(title_id))) != QMessageBox::Yes) {
        return;
    }
    Finish(QtCommon::ModList::Remove(*mod, title_id), tr("Removed %1.").arg(name), title_id);
}

void ModListDialog::OpenModsFolder() {
    u64 title_id = 0;
    auto dir = Common::FS::GetEdenPath(Common::FS::EdenPath::LoadDir);
    if (SelectedMod(&title_id) != nullptr) {
        dir /= fmt::format("{:016X}", title_id);
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    QDesktopServices::openUrl(
        QUrl::fromLocalFile(QString::fromStdString(Common::FS::PathToUTF8String(dir))));
}
