// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <filesystem>
#include <optional>
#include <system_error>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/net/net.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "qt_common/util/compress.h"
#include "qt_common/util/mod_list.h"

namespace QtCommon::ModList {
namespace {

namespace fs = std::filesystem;

constexpr const char* IndexName = "mods.json";
// Written into each installed mod: {"id": ..., "version": ...}.
constexpr const char* MarkerName = ".modlist.json";

// https://github.com/<owner>/<repo>[.git][/tree/<branch>] -> the path of its zip archive on
// github.com. Without a branch, HEAD is the repository's default branch.
std::optional<std::string> ArchivePath(std::string url) {
    constexpr std::string_view prefix = "https://github.com/";
    // Configs saved before string settings kept a URL's "//" hold "https:/github.com/...".
    if (url.starts_with("https:/") && !url.starts_with("https://")) {
        url.insert(6, "/");
    }
    if (!url.starts_with(prefix)) {
        return std::nullopt;
    }
    std::vector<std::string> parts;
    Common::SplitString(url.substr(prefix.size()), '/', parts);
    std::erase(parts, std::string{});
    if (parts.size() < 2) {
        return std::nullopt;
    }
    auto repo = parts[1];
    if (repo.ends_with(".git")) {
        repo.resize(repo.size() - 4);
    }
    if (parts.size() >= 4 && parts[2] == "tree") {
        return fmt::format("/{}/{}/archive/refs/heads/{}.zip", parts[0], repo, parts[3]);
    }
    return fmt::format("/{}/{}/archive/HEAD.zip", parts[0], repo);
}

fs::path ToPath(const QString& path) {
    return fs::path{path.toStdU16String()};
}

QString ToQString(const fs::path& path) {
    return QString::fromStdU16String(path.u16string());
}

// The unpacked repository from the last refresh.
fs::path RepoDir() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "mod_list" / "repo";
}

// A mod's folder must stay inside the repository.
bool StaysInside(const QString& relative) {
    const auto normal = ToPath(relative).lexically_normal();
    return !relative.isEmpty() && !normal.is_absolute() && !normal.has_root_name() &&
           *normal.begin() != "..";
}

QString String(const QJsonObject& object, const char* key) {
    return object.value(QLatin1String(key)).toVariant().toString();
}

Index ParseIndex(const fs::path& repo) {
    Index index;
    QFile file{ToQString(repo / IndexName)};
    if (!file.open(QIODevice::ReadOnly)) {
        index.error = QStringLiteral("The repository has no mods.json.");
        return index;
    }
    QJsonParseError parse_error{};
    const auto document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (!document.isObject()) {
        index.error = QStringLiteral("mods.json can't be read: %1").arg(parse_error.errorString());
        return index;
    }
    for (const auto& value : document.object().value(QStringLiteral("mods")).toArray()) {
        const auto object = value.toObject();
        Mod mod{.id = String(object, "id"),
                .name = String(object, "name"),
                .title_ids = {},
                .version = String(object, "version"),
                .author = String(object, "author"),
                .category = String(object, "category"),
                .description = String(object, "description"),
                .path = String(object, "path")};
        for (const auto& title : object.value(QStringLiteral("title_ids")).toArray()) {
            auto text = title.toString();
            if (text.startsWith(QStringLiteral("0x"), Qt::CaseInsensitive)) {
                text.remove(0, 2);
            }
            bool ok = false;
            const u64 title_id = text.toULongLong(&ok, 16);
            if (ok && title_id != 0) {
                mod.title_ids.push_back(title_id);
            }
        }
        if (mod.id.isEmpty() || mod.title_ids.empty() || !StaysInside(mod.path) ||
            mod.FolderName().isEmpty()) {
            LOG_WARNING(Frontend, "mods.json: entry '{}' needs an id, title_ids and a path",
                        mod.id.toStdString());
            continue;
        }
        if (mod.name.isEmpty()) {
            mod.name = mod.FolderName();
        }
        index.mods.push_back(std::move(mod));
    }
    return index;
}

struct Marker {
    QString id;
    QString version;
};

std::optional<Marker> ReadMarker(const fs::path& dir) {
    QFile file{ToQString(dir / MarkerName)};
    if (!file.open(QIODevice::ReadOnly)) {
        return std::nullopt;
    }
    const auto object = QJsonDocument::fromJson(file.readAll()).object();
    return Marker{String(object, "id"), String(object, "version")};
}

bool WriteMarker(const fs::path& dir, const Mod& mod) {
    QFile file{ToQString(dir / MarkerName)};
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return false;
    }
    const QJsonObject object{{QStringLiteral("id"), mod.id}, {QStringLiteral("version"), mod.version}};
    return file.write(QJsonDocument{object}.toJson()) > 0;
}

QString ExceptionMessage(const char* what) {
    return QStringLiteral("Unexpected error: %1").arg(QString::fromUtf8(what));
}

} // Anonymous namespace

QString Mod::FolderName() const {
    return path.section(QLatin1Char('/'), -1, -1, QString::SectionSkipEmpty);
}

Index Refresh(const std::string& repo_url) {
    Index index;
    try {
        const auto archive_path = ArchivePath(Common::StripSpaces(repo_url));
        if (!archive_path) {
            index.error = QStringLiteral("%1 is not a GitHub repository URL.")
                              .arg(QString::fromStdString(repo_url));
            return index;
        }

        // MakeRequest follows redirects: github.com sends archives to codeload.github.com, and a
        // renamed repository to its new name.
        LOG_INFO(Frontend, "Downloading the mod list from https://github.com{}", *archive_path);
        const auto body = Common::Net::MakeRequest("https://github.com", *archive_path);
        if (!body || body->empty()) {
            index.error =
                QStringLiteral("Could not download %1.").arg(QString::fromStdString(repo_url));
            return index;
        }

        QTemporaryDir temp;
        if (!temp.isValid()) {
            index.error = QStringLiteral("Could not create a temporary folder.");
            return index;
        }
        const QString zip_path = temp.filePath(QStringLiteral("repo.zip"));
        QFile zip{zip_path};
        if (!zip.open(QIODevice::WriteOnly) ||
            zip.write(body->data(), static_cast<qint64>(body->size())) !=
                static_cast<qint64>(body->size())) {
            index.error = QStringLiteral("Could not save the download.");
            return index;
        }
        zip.close();

        const QString unpacked = temp.filePath(QStringLiteral("unpacked"));
        if (Compress::extractDir(zip_path, unpacked).isEmpty()) {
            index.error = QStringLiteral("The download is not a readable zip archive.");
            return index;
        }

        // GitHub archives hold one folder, <repo>-<branch or commit>.
        fs::path repo_root = ToPath(unpacked);
        std::error_code ec;
        std::vector<fs::path> entries;
        for (const auto& entry : fs::directory_iterator{repo_root, ec}) {
            entries.push_back(entry.path());
        }
        if (entries.size() == 1 && fs::is_directory(entries.front(), ec)) {
            repo_root = entries.front();
        }

        // Keep the previous list unless the new one reads.
        const auto fresh = ParseIndex(repo_root);
        if (!fresh.error.isEmpty()) {
            return fresh;
        }
        const auto cache = RepoDir();
        fs::remove_all(cache, ec);
        fs::create_directories(cache.parent_path(), ec);
        fs::copy(repo_root, cache, fs::copy_options::recursive, ec);
        if (ec) {
            index.error = QStringLiteral("Could not store the mod list: %1")
                              .arg(QString::fromStdString(ec.message()));
            return index;
        }
        return ParseIndex(cache);
    } catch (const std::exception& e) {
        LOG_ERROR(Frontend, "Mod list refresh failed: {}", e.what());
        index.error = ExceptionMessage(e.what());
    } catch (...) {
        index.error = ExceptionMessage("unknown");
    }
    return index;
}

Index LoadCached() {
    try {
        std::error_code ec;
        if (!fs::exists(RepoDir() / IndexName, ec)) {
            return {};
        }
        return ParseIndex(RepoDir());
    } catch (const std::exception& e) {
        return {.mods = {}, .error = ExceptionMessage(e.what())};
    }
}

fs::path GetInstallDir(const Mod& mod, u64 title_id) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LoadDir) /
           fmt::format("{:016X}", title_id) / ToPath(mod.FolderName());
}

State GetState(const Mod& mod, u64 title_id) {
    std::error_code ec;
    const auto dir = GetInstallDir(mod, title_id);
    if (!fs::exists(dir, ec)) {
        return State::NotInstalled;
    }
    const auto marker = ReadMarker(dir);
    if (!marker || marker->id != mod.id) {
        return State::Foreign;
    }
    return marker->version == mod.version ? State::Installed : State::UpdateAvailable;
}

QString Install(const Mod& mod, u64 title_id) {
    try {
        std::error_code ec;
        const auto source = RepoDir() / ToPath(mod.path);
        if (!fs::is_directory(source, ec)) {
            return QStringLiteral("%1 isn't in the downloaded list. Refresh the list and try again.")
                .arg(mod.name);
        }
        const auto dir = GetInstallDir(mod, title_id);
        switch (GetState(mod, title_id)) {
        case State::Foreign:
            return QStringLiteral("A mod folder named \"%1\" already exists. Remove or rename it "
                                  "first.")
                .arg(mod.FolderName());
        case State::Installed:
        case State::UpdateAvailable:
            fs::remove_all(dir, ec);
            break;
        case State::NotInstalled:
            break;
        }
        fs::create_directories(dir.parent_path(), ec);
        fs::copy(source, dir, fs::copy_options::recursive, ec);
        if (ec || !WriteMarker(dir, mod)) {
            fs::remove_all(dir, ec);
            return QStringLiteral("Could not copy %1 into %2.").arg(mod.name, ToQString(dir));
        }

        // Shown enabled in the Add-Ons tab.
        auto& disabled = Settings::values.disabled_addons[title_id];
        std::erase(disabled, mod.FolderName().toStdString());
        return {};
    } catch (const std::exception& e) {
        return ExceptionMessage(e.what());
    }
}

QString Remove(const Mod& mod, u64 title_id) {
    try {
        switch (GetState(mod, title_id)) {
        case State::NotInstalled:
            return {};
        case State::Foreign:
            return QStringLiteral("\"%1\" wasn't installed from the mod list, so it's left alone.")
                .arg(mod.FolderName());
        case State::Installed:
        case State::UpdateAvailable:
            break;
        }
        std::error_code ec;
        fs::remove_all(GetInstallDir(mod, title_id), ec);
        if (ec) {
            return QStringLiteral("Could not remove %1: %2")
                .arg(mod.name, QString::fromStdString(ec.message()));
        }
        return {};
    } catch (const std::exception& e) {
        return ExceptionMessage(e.what());
    }
}

} // namespace QtCommon::ModList
