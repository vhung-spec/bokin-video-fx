"""Keyboard shortcuts: every action a key does on the page, its keys out of the box, and the keys the owner chose.

The page (index.html) reads the list and the keys in use from /api/keys and does an action when one of its keys is
pressed; its "Phím tắt" window changes them (POST /api/keys). A key is written as the KeyboardEvent.code of the key
(its place on the keyboard, the same with any layout or Vietnamese typing tool: "KeyZ", "Digit1", "Equal", "Space")
after the modifiers held with it, in this order: "Ctrl+Shift+KeyZ".

Only the actions whose keys the owner changed are kept (keys.json in the data folder), so the defaults of a later
version still reach every action left as it was; a default that one of the owner's keys already uses is dropped.

Where an action works: "all" (always), "edit" (not while painting an area), "paint" (only while painting). Two
actions may share a key only when they never work at the same time: one "edit", the other "paint".

HUONG-DAN.md (the owner's guide) lists the defaults in a block written by markdown(): after changing the list,
`python -m fx.keys` writes it there again (a test checks the guide has the current one).
"""
import json
import os
import re
import string

from . import paths

GROUPS = ["Phát và xem", "Hình", "Timeline", "Vẽ vùng", "Chung"]
GROUP_NOTES = {"Vẽ vùng": "chỉ khi đang vẽ vùng (hai phím Khép và Bỏ điểm cũng dùng khi vẽ dây đèn). Lúc vẽ vùng, các "
                          "phím chỉ dùng ngoài lúc vẽ (Phát, Dao cắt…) không chạy"}


def _a(id_, group, label, keys, where="all"):
    return {"id": id_, "group": group, "label": label, "keys": keys, "where": where}


ACTIONS = [
    _a("playPause", "Phát và xem", "Phát / dừng. Giữ phím rồi kéo chuột: dời hình", ["Space"]),
    _a("play", "Phát và xem", "Phát", ["KeyL"], "edit"),
    _a("pause", "Phát và xem", "Dừng", ["KeyK"], "edit"),
    _a("stepBack", "Phát và xem", "Lùi một khung", ["ArrowLeft"]),
    _a("stepFwd", "Phát và xem", "Tới một khung", ["ArrowRight"]),
    _a("stepBack5", "Phát và xem", "Lùi 5 khung", ["Shift+ArrowLeft"]),
    _a("stepFwd5", "Phát và xem", "Tới 5 khung", ["Shift+ArrowRight"]),
    _a("prevEdit", "Phát và xem", "Về mép clip ngay trước", ["ArrowUp"]),
    _a("nextEdit", "Phát và xem", "Tới mép clip ngay sau", ["ArrowDown"]),
    _a("toStart", "Phát và xem", "Về đầu video", ["Home"]),
    _a("toEnd", "Phát và xem", "Về cuối video", ["End"]),
    _a("picIn", "Hình", "Phóng to hình (quanh chỗ chuột)", ["Ctrl+Equal", "Ctrl+NumpadAdd"]),
    _a("picOut", "Hình", "Thu nhỏ hình (quanh chỗ chuột)", ["Ctrl+Minus", "Ctrl+NumpadSubtract"]),
    _a("picFit", "Hình", "Hình vừa khung, ở giữa", ["Ctrl+Digit0", "Ctrl+Numpad0"]),
    _a("pic100", "Hình", "Hình cỡ thật (100%)", ["Ctrl+Digit1", "Ctrl+Numpad1"]),
    _a("zoomIn", "Hình", "Phóng to chỗ chuột đang ở (hình hoặc timeline)", ["Equal", "NumpadAdd"], "edit"),
    _a("zoomOut", "Hình", "Thu nhỏ chỗ chuột đang ở (hình hoặc timeline)", ["Minus", "NumpadSubtract"], "edit"),
    _a("zoomFit", "Hình", "Chuột trên hình: vừa khung. Chuột trên timeline: xem cả timeline", ["Backslash"], "edit"),
    _a("hand", "Hình", "Bàn tay: kéo để dời hình. Giữ phím mà kéo thì thả phím là thôi", ["KeyH"]),
    _a("selectTool", "Timeline", "Công cụ chọn (cũng thôi Bàn tay)", ["KeyV"], "edit"),
    _a("razorTool", "Timeline", "Dao cắt: bấm vào clip để cắt đôi", ["KeyC"], "edit"),
    _a("snap", "Timeline", "Bật / tắt bám dính (cả lúc đang kéo clip)", ["KeyS"], "edit"),
    _a("split", "Timeline", "Cắt clip tại đầu phát", ["Ctrl+KeyK"]),
    _a("deleteClip", "Timeline", "Xoá clip đang chọn (đang vẽ dây đèn, đặt ống khói: xoá dây, ống khói đang chọn)",
       ["Delete", "Backspace"], "edit"),
    _a("copyClip", "Timeline", "Sao chép clip đang chọn (dán được sang track khác, hoặc dự án khác)", ["Ctrl+KeyC"], "edit"),
    _a("pasteClip", "Timeline", "Dán clip đã chép: tại đầu phát, vào track dưới chuột", ["Ctrl+KeyV"], "edit"),
    _a("pasteAttrs", "Timeline", "Dán thiết lập của clip đã chép (màu, cỡ, vị trí…) vào clip đang chọn",
       ["Ctrl+Alt+KeyV"], "edit"),
    _a("undo", "Timeline", "Hoàn tác (đang khoanh: bỏ điểm vừa bấm)", ["Ctrl+KeyZ"]),
    _a("redo", "Timeline", "Làm lại", ["Ctrl+KeyY", "Ctrl+Shift+KeyZ"]),
    _a("brushTool", "Vẽ vùng", "Cọ", ["KeyB"], "paint"),
    _a("polyTool", "Vẽ vùng", "Khoanh", ["KeyL"], "paint"),
    _a("brushMode", "Vẽ vùng", "Đổi Thêm / Bớt", ["KeyX"], "paint"),
    _a("brushSmaller", "Vẽ vùng", "Cọ nhỏ lại", ["BracketLeft"], "paint"),
    _a("brushBigger", "Vẽ vùng", "Cọ to lên", ["BracketRight"], "paint"),
    _a("polyClose", "Vẽ vùng", "Khép vùng đang khoanh (vẽ dây đèn: xong dây đang vẽ)", ["Enter"], "paint"),
    _a("polyBack", "Vẽ vùng", "Bỏ điểm vừa khoanh (vẽ dây đèn: bỏ điểm cuối)", ["Backspace"], "paint"),
    _a("save", "Chung", "Lưu dự án (app cũng tự lưu mỗi lần chỉnh)", ["Ctrl+KeyS"]),
    _a("export", "Chung", "Xuất video", ["Ctrl+KeyM"]),
    _a("keys", "Chung", "Mở bảng Phím tắt và hướng dẫn", ["Shift+Slash"]),
]
BY_ID = {a["id"]: a for a in ACTIONS}

# Keys that always do the same thing (not in the list above, so not changeable)
FIXED = [
    ("Esc", "Thôi: bỏ chọn clip, thôi Bàn tay, bỏ đường đang khoanh, thôi vẽ vùng, xong dây đèn đang vẽ (lần nữa: thôi "
            "vẽ dây), thôi đặt ống khói, đóng cửa sổ đang mở"),
    ("Enter", "Trong ô số: giữ số vừa gõ"),
    ("Tab", "Chuyển sang nút hoặc ô kế tiếp"),
]

# What each key is called on the page and in the guide
NAMES = {
    **{"Key" + c: c for c in string.ascii_uppercase},
    **{"Digit" + d: d for d in string.digits},
    **{"Numpad" + d: "Num " + d for d in string.digits},
    "NumpadAdd": "Num +", "NumpadSubtract": "Num -", "NumpadMultiply": "Num *", "NumpadDivide": "Num /",
    "NumpadDecimal": "Num .", "NumpadEnter": "Num Enter",
    **{f"F{n}": f"F{n}" for n in range(1, 13)},
    "ArrowLeft": "←", "ArrowRight": "→", "ArrowUp": "↑", "ArrowDown": "↓",
    "Home": "Home", "End": "End", "PageUp": "Page Up", "PageDown": "Page Down", "Insert": "Insert",
    "Delete": "Delete", "Backspace": "Backspace", "Enter": "Enter", "Space": "Phím cách",
    "Equal": "=", "Minus": "-", "BracketLeft": "[", "BracketRight": "]", "Backslash": "\\", "Semicolon": ";",
    "Quote": "'", "Comma": ",", "Period": ".", "Slash": "/", "Backquote": "`",
}
# Windows' and the browser's own keys: a page cannot stop them (closing the window, a new window, reloading,
# the browser's menus and tools), or pressing them does that too
RESERVED = {"Ctrl+KeyW", "Ctrl+KeyN", "Ctrl+KeyT", "Ctrl+KeyQ", "Ctrl+KeyR", "Ctrl+Shift+KeyW", "Ctrl+Shift+KeyN",
            "Ctrl+Shift+KeyT", "Ctrl+Shift+KeyR", "Ctrl+Shift+KeyI", "Ctrl+Shift+KeyJ", "Ctrl+Shift+KeyC",
            "Ctrl+Shift+KeyB", "Ctrl+Shift+KeyO", "Ctrl+Shift+Delete", "Alt+F4", "Ctrl+F4", "Alt+Space",
            "Alt+KeyF", "Alt+KeyE", "Alt+Home", "Alt+ArrowLeft", "Alt+ArrowRight"}
RESERVED_KEYS = {"F1", "F3", "F5", "F6", "F7", "F10", "F11", "F12"}  # with any modifiers
MAX_KEYS = 4  # per action
COMBO = re.compile(r"^(Ctrl\+)?(Alt\+)?(Shift\+)?([A-Za-z0-9]{1,20})$")
FILE = "keys.json"
GUIDE = "HUONG-DAN.md"
BLOCK = ("<!-- keys: do `python -m fx.keys` viết ra, đừng sửa tay -->", "<!-- /keys -->")


def usable(combo):
    """A key the page may give an action: one it knows, not Windows' or the browser's own."""
    m = COMBO.match(combo) if isinstance(combo, str) else None
    return bool(m) and m.group(4) in NAMES and m.group(4) not in RESERVED_KEYS and combo not in RESERVED


def clash(a, b):
    """Two actions that may work at the same time (so cannot share a key)."""
    return a == b or "all" in (a, b)


def clean(data):
    """The owner's keys as sent by the page or read back: known actions, usable keys, at most MAX_KEYS each, none
    used twice where both actions work (the first action in the list keeps it)."""
    if not isinstance(data, dict):
        return {}
    out, taken = {}, []
    for a in ACTIONS:
        keys = data.get(a["id"])
        if not isinstance(keys, list):
            continue
        mine = []
        for k in keys:
            if usable(k) and k not in mine and len(mine) < MAX_KEYS \
                    and not any(k == t and clash(a["where"], w) for t, w in taken):
                mine.append(k)
        out[a["id"]] = mine
        taken += [(k, a["where"]) for k in mine]
    return out


def effective(chosen):
    """The keys of every action: the owner's where chosen, else the defaults no chosen key already uses."""
    taken = [(k, BY_ID[i]["where"]) for i, keys in chosen.items() for k in keys]
    return {a["id"]: list(chosen[a["id"]]) if a["id"] in chosen else
            [k for k in a["keys"] if not any(k == t and clash(a["where"], w) for t, w in taken)] for a in ACTIONS}


def load():
    try:
        with open(os.path.join(paths.data_dir(), FILE), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return clean(data.get("keys") if isinstance(data, dict) else None)


def save(chosen):
    """Keeps the owner's keys (only the actions that differ from the defaults); returns what was kept."""
    chosen = {i: k for i, k in clean(chosen).items() if k != BY_ID[i]["keys"]}
    path = os.path.join(paths.data_dir(), FILE)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"keys": chosen}, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return chosen


def view():
    """What the page needs: the actions, the names of keys, the keys in use and which the owner changed."""
    chosen = load()
    return {"actions": ACTIONS, "groups": GROUPS, "group_notes": GROUP_NOTES, "fixed": FIXED, "names": NAMES,
            "reserved": sorted(RESERVED), "reserved_keys": sorted(RESERVED_KEYS), "max": MAX_KEYS,
            "keys": effective(chosen), "custom": chosen}


def name(combo):
    """'Ctrl+Shift+KeyZ' -> 'Ctrl+Shift+Z'."""
    *mods, code = combo.split("+")
    return "+".join(mods + [NAMES.get(code, code)])


def _code(text):
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def markdown():
    """The guide's block of default keys (HUONG-DAN.md, between the BLOCK lines)."""
    lines = [BLOCK[0]]
    for g in GROUPS:
        note = GROUP_NOTES.get(g)
        lines += ["", f"### {g}", ""] + ([f"Phím nhóm này {note}.", ""] if note else [])
        lines += ["| Việc | Phím |", "|---|---|"]
        lines += [f"| {a['label']} | {' hoặc '.join(_code(name(k)) for k in a['keys'])} |"
                  for a in ACTIONS if a["group"] == g]
    lines += ["", "### Phím cố định (không đổi được)", "", "| Phím | Việc |", "|---|---|"]
    lines += [f"| {_code(k)} | {what} |" for k, what in FIXED]
    return "\n".join(lines + ["", BLOCK[1]])


def guide_path():
    """HUONG-DAN.md: in the bundle next to index.html, or at the top of the source folder."""
    here = os.path.join(paths.resource_dir(), GUIDE)
    return here if os.path.exists(here) else os.path.join(os.path.dirname(paths.resource_dir()), GUIDE)


def with_block(text):
    """The guide's text with the current block of default keys."""
    head, rest = text.split(BLOCK[0], 1)
    return head + markdown() + rest.split(BLOCK[1], 1)[1]


if __name__ == "__main__":  # python -m fx.keys: the guide's block again, after the list changed
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), GUIDE)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(with_block(text))
    print("HUONG-DAN.md: phím mặc định đã ghi lại")
