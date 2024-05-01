ACTION_MAPPING = {
    "a": "KEY_A",
    "b": "KEY_B",
    "c": "KEY_C",
    "d": "KEY_D",
    "e": "KEY_E",
    "f": "KEY_F",
    "g": "KEY_G",
    "h": "KEY_H",
    "i": "KEY_I",
    "j": "KEY_J",
    "k": "KEY_K",
    "l": "KEY_L",
    "m": "KEY_M",
    "n": "KEY_N",
    "o": "KEY_O",
    "p": "KEY_P",
    "q": "KEY_Q",
    "r": "KEY_R",
    "s": "KEY_S",
    "t": "KEY_T",
    "u": "KEY_U",
    "v": "KEY_V",
    "w": "KEY_W",
    "x": "KEY_X",
    "y": "KEY_Y",
    "z": "KEY_Z",
    "1": "KEY_1",
    "2": "KEY_2",
    "3": "KEY_3",
    "4": "KEY_4",
    "5": "KEY_5",
    "6": "KEY_6",
    "7": "KEY_7",
    "8": "KEY_8",
    "9": "KEY_9",
    "0": "KEY_0",
    "!": "KEY_1",
    "@": "KEY_2",
    "#": "KEY_3",
    "$": "KEY_4",
    "¥": "KEY_4",
    "%": "KEY_5",
    "^": "KEY_6",
    "&": "KEY_7",
    "*": "KEY_8",
    "(": "KEY_9",
    ")": "KEY_0",
    "（": "KEY_9",
    "）": "KEY_0",
    "-": "KEY_MINUS",
    "——": "KEY_MINUS",
    "_": "KEY_MINUS",
    "=": "KEY_EQUAL",
    "+": "KEY_EQUAL",
    '"': "KEY_QUOTE",
    "“": "KEY_QUOTE",
    "”": "KEY_QUOTE",
    "'": "KEY_QUOTE",
    "‘": "KEY_QUOTE",
    "’": "KEY_QUOTE",
    ":": "KEY_SEMICOLON",
    "：": "KEY_SEMICOLON",
    ";": "KEY_SEMICOLON",
    "；": "KEY_SEMICOLON",
    "[": "KEY_LEFTBRACE",
    "{": "KEY_LEFTBRACE",
    "]": "KEY_RIGHTBRACE",
    "}": "KEY_RIGHTBRACE",
    "【": "KEY_LEFTBRACE",
    "「": "KEY_LEFTBRACE",
    "】": "KEY_RIGHTBRACE",
    "」": "KEY_RIGHTBRACE",
    ",": "KEY_COMMA",
    "，": "KEY_COMMA",
    "<": "KEY_COMMA",
    "《": "KEY_COMMA",
    ".": "KEY_DOT",
    "。": "KEY_DOT",
    ">": "KEY_DOT",
    "》": "KEY_DOT",
    "/": "KEY_SLASH",
    "、": "KEY_SLASH",
    "?": "KEY_SLASH",
    "？": "KEY_SLASH",
    "\\": "KEY_BACKSLASH",
    "|": "KEY_BACKSLASH",
    "、": "KEY_GRAVE",
    "`": "KEY_GRAVE",
    "~": "KEY_GRAVE",
    "·": "KEY_GRAVE",
    "Key.space": "KEY_SPACE",
    "Key.shift": "KEY_SHIFT",
    "Key.ctrl": "KEY_CTRL",
    "Key.alt": "KEY_ALT",
    "Key.backspace": "KEY_BACKSPACE",
    "Key.enter": "KEY_ENTER",
    "Key.tab": "KEY_TAB",
    "Key.up": "KEY_UP",
    "Key.down": "KEY_DOWN",
    "Key.left": "KEY_LEFT",
    "Key.right": "KEY_RIGHT",
    "Key.esc": "KEY_ESC",
    "Key.cmd": "KEY_CMD",
    "Key.delete": "KEY_BACKSPACE",
    "Key.caps_lock": "KEY_CAPSLOCK",
    "Button.left": "MOUSE_LEFT_SINGLE",
    "DOUBLE_CLICK": "MOUSE_LEFT_DOUBLE",
    "Button.right": "MOUSE_RIGHT",
    "Button.middle": "MOUSE_MIDDLE",
    "drag_start": "DRAG_START",
    "drag_end": "DRAG_END",
    "scroll": "SCROLL",
    "NO_ACTION": "NO_ACTION",
}

def action_num():
    return len(set(ACTION_MAPPING))

def action_types():
    return list(set(ACTION_MAPPING))