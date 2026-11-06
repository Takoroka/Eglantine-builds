# Mod List

**Tools > Mod List** shows the community mods for the games in your game list.
- **Installing:** a mod is copied into `load/<title id>/<mod name>/`, where the game's Add-Ons tab
  enables and disables it like any other mod.
- **Nothing is fetched on its own:** the list is downloaded when you click **Refresh list**, and
  installs use that download.

## The window

- **Grouped by game:** mods are listed under each game the game list found (by title ID). Each shows
  its version, author and status:
  - **Not installed**;
  - **Installed**;
  - **Update available** (the list has a different version);
  - **Folder exists** (a mod folder with that name that the list didn't install; the list leaves it
    alone).
- **Install / Update:** copies the mod and enables it for that game.
- **Remove:** deletes a mod the list installed.
- **Refresh list:** downloads the mod repository again.
- **Open mods folder:** opens `load/<title id>/` for the selected game, or `load/`.

The repository is the `mod_list_repo_url` setting, shown in the window. By default it is
https://github.com/Takoroka/Eglantine-Mods. GitHub redirects renamed repositories, and the
download follows the redirect.

## The repository

A GitHub repository with `mods.json` at its root, and one folder per mod laid out exactly as
`load/<title id>/<mod name>/` expects (`exefs/`, `romfs/`, `cheats/`, ...):

```
mods.json
mods/0400C3F00006E000/Mario Kart World - 30 FPS/exefs/<build id>.ips
```

```json
{
  "format": 1,
  "mods": [
    {
      "id": "mkw-30fps",
      "name": "Mario Kart World - 30 FPS",
      "title_ids": ["0400C3F00006E000"],
      "version": "1",
      "author": "Franz (FZout)",
      "category": "Performance",
      "description": "Always use the game's native 30 FPS mode.",
      "path": "mods/0400C3F00006E000/Mario Kart World - 30 FPS"
    }
  ]
}
```

| Field | |
|---|---|
| `id` | Required. Unique and stable: it is how an installed mod is recognized. |
| `title_ids` | Required. Program IDs in hex. |
| `path` | Required. The mod's folder in the repository. Its last part is the installed folder name, and the name the Add-Ons tab shows. |
| `name`, `version`, `author`, `category`, `description` | Shown in the window. A changed `version` shows as an update. |

- **Install marker:** each installed mod gets a `.modlist.json` with its `id` and `version`; that's
  how the window tells installed, updated and foreign folders apart.
- **What a mod can hold:** whatever the mod loader already supports: exefs patches (IPS and IPS32
  named by the module's build ID, IPSwitch `.pchtxt`), exefs file replacements, `romfs/` and
  `cheats/`.
- **Game versions:** exefs patches apply only to the game version whose build ID they're named for,
  so a mod's description should say which version it's for.
