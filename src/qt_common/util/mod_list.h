// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <QString>

#include "common/common_types.h"

// The community mod list: a GitHub repository with a mods.json index and the mods' folders, laid
// out as load/<title id>/<mod>/ expects (docs/ModList.md). Nothing is fetched unless asked.
namespace QtCommon::ModList {

struct Mod {
    QString id;
    QString name;
    std::vector<u64> title_ids;
    QString version;
    QString author;
    QString category;
    QString description;
    QString path; // The mod's folder in the repository; its last part is the installed name.

    [[nodiscard]] QString FolderName() const;
};

struct Index {
    std::vector<Mod> mods;
    QString error; // Empty on success.
};

enum class State {
    NotInstalled,
    Installed,
    UpdateAvailable,
    Foreign, // A folder with the mod's name that the list didn't install.
};

// Downloads the repository at repo_url (https://github.com/<owner>/<repo>, optionally
// /tree/<branch>) into the cache and reads its mods.json. Follows GitHub's redirects, so a renamed
// repository keeps working.
[[nodiscard]] Index Refresh(const std::string& repo_url);

// The index from the last refresh, without the network; empty before the first one.
[[nodiscard]] Index LoadCached();

// load/<title id>/<mod folder>
[[nodiscard]] std::filesystem::path GetInstallDir(const Mod& mod, u64 title_id);

[[nodiscard]] State GetState(const Mod& mod, u64 title_id);

// Copies the mod from the last refreshed repository into its install folder and enables it for
// the title. Returns an error message, empty on success.
[[nodiscard]] QString Install(const Mod& mod, u64 title_id);

// Deletes a mod the list installed. Returns an error message, empty on success.
[[nodiscard]] QString Remove(const Mod& mod, u64 title_id);

} // namespace QtCommon::ModList
