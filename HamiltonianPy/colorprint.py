"""
ANSI Color formatting for output in terminal


See also https://en.wikipedia.org/wiki/ANSI_escape_code
"""

from __future__ import print_function

__all__ = ["colored", "cprint", "cdemo"]

COLOR_NAMES_STD_LONG = (
        "black", "red", "green", "yellow","blue", "magenta", "cyan", "white")
COLOR_NAMES_STD_SHORT = ('k', 'r', 'g', 'y', 'b', 'm', 'c', 'w')
SHORT2LONG = dict(zip(COLOR_NAMES_STD_SHORT, COLOR_NAMES_STD_LONG))

COLOR_CODES_STD_4BITS_FG = (30, 31, 32, 33, 34, 35, 36, 37)
COLOR_CODES_STD_4BITS_BG = (40, 41, 42, 43, 44, 45, 46, 47)
COLOR_CODES_STD_8BITS = (0, 1, 2, 3, 4, 5, 6, 7)
COLOR_CODES_STD_24BITS = (
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    )

COLOR_CODES_ALL_4BITS_FG = tuple(range(30, 38)) + tuple(range(90, 98))
COLOR_CODES_ALL_4BITS_BG = tuple(range(40, 48)) + tuple(range(100, 108))
COLOR_CODES_ALL_8BITS = tuple(range(256))
COLOR_CODES_ALL_24BITS_R = COLOR_CODES_ALL_8BITS
COLOR_CODES_ALL_24BITS_G = COLOR_CODES_ALL_8BITS
COLOR_CODES_ALL_24BITS_B = COLOR_CODES_ALL_8BITS

NAME2CODE_4BITS_FG = dict(zip(COLOR_NAMES_STD_LONG, COLOR_CODES_STD_4BITS_FG))
NAME2CODE_4BITS_BG = dict(zip(COLOR_NAMES_STD_LONG, COLOR_CODES_STD_4BITS_BG))
NAME2CODE_8BITS = dict(zip(COLOR_NAMES_STD_LONG, COLOR_CODES_STD_8BITS))
NAME2CODE_24BITS = dict(zip(COLOR_NAMES_STD_LONG, COLOR_CODES_STD_24BITS))

CSI= "\x1B["
RESET = CSI + "0m"

# Return the color name according to the given 'color' parameter
def _color_name(color):# {{{
    tmp = color.lower()
    if tmp in COLOR_NAMES_STD_LONG:
        color_name = tmp
    elif tmp in COLOR_NAMES_STD_SHORT:
        color_name = SHORT2LONG[tmp]
    else:
        raise ValueError("Unrecognized color specifier.")
    return color_name
# }}}

# Return the color-control-sequence according to the given color parameter
# The supported color scheme is 4-bits, 8-bits, 24-bits(rgb)
# See wikipedia page: https://en.wikipedia.org/wiki/ANSI_escape_code for detail
# The 'fg' parameter of these _color_ctrl_seq_* functions determines whether to
# preform on foreground or background
# If fg set to True, these functions contrl the foreground color
# If fg set to False, these functions contrl the background color

def _color_ctrl_seq_4bits(color=None, fg=True):# {{{
    if isinstance(color, int):
        supported = COLOR_CODES_ALL_4BITS_FG if fg else COLOR_CODES_ALL_4BITS_BG
        if color in supported:
            color_code = color
        else:
            raise ValueError("The given color code is not supported by "
                    "the 4-bits color scheme.")
    elif isinstance(color, str):
        name2code = NAME2CODE_4BITS_FG if fg else NAME2CODE_4BITS_BG
        color_code = name2code[_color_name(color)]
    else:
        return ""
    return CSI + "{}m".format(color_code)
# }}}

def _color_ctrl_seq_8bits(color=None, fg=True):# {{{
    if isinstance(color, int):
        if color in COLOR_CODES_ALL_8BITS:
            color_code = color
        else:
            raise ValueError("The given color code is not supported by "
                    "the 8-bits color scheme.")
    elif isinstance(color, str):
        color_code = NAME2CODE_8BITS[_color_name(color)]
    else:
        return ""
    fmt = CSI + "{0};5;{1}m"
    return fmt.format(38, color_code) if fg else fmt.format(48, color_code)
# }}}

def _color_ctrl_seq_24bits(color=None, fg=True):# {{{
    if isinstance(color, (list, tuple)) and len(color) == 3:
        r, g, b = color
        if not (r in COLOR_CODES_ALL_24BITS_R and
                g in COLOR_CODES_ALL_24BITS_G and
                b in COLOR_CODES_ALL_24BITS_B):
            raise ValueError("The given color code is not supported by "
                    "the 24-bits color scheme.")
    elif isinstance(color, str):
        if color.startswith('#') and len(color) == 7:
            r = int(color[1:3], base=16)
            g = int(color[3:5], base=16)
            b = int(color[5:7], base=16)
        else:
            r, g, b = NAME2CODE_24BITS[_color_name(color)]
    else:
        return ""
    fmt = CSI + "{0};2;{r};{g};{b}m"
    return fmt.format(38,r=r,g=g,b=b) if fg else fmt.format(48,r=r,g=g,b=b)
# }}}

def colored(string, fg=None, bg=None, bold=False, color_scheme=8):# {{{
    """
    Decorate the given string with ANSI color control sequence

    Parameters
    ----------
    string : str
        The string to be decorated
    fg : None | str | int | tuple | list, optional, keyword-only
        Describe the foreground color of the string
        default : None
    bg : None | str | int | tuple | list, optional, keyword-only
        Describe the background color of the string
        default : None
    bold : boolean, optional, keyword-only
        Whether to use bold font
        default : False
    color_scheme : 4 | 8 | 24 | "rgb", keyword-only
        which color scheme to use
        "rgb" is the same as 24.
        default : 8

    Returns
    -------
    res : str
        The decorated string

    See also
    --------
    https://en.wikipedia.org/wiki/ANSI_escape_code
    cprint
    """

    if bold:
        bold_ctrl_seq = CSI + "1m"
    else:
        bold_ctrl_seq = ""

    if color_scheme == 4:
        color_ctrl_seq = _color_ctrl_seq_4bits
    elif color_scheme == 8:
        color_ctrl_seq = _color_ctrl_seq_8bits
    elif color_scheme == 24 or color_scheme == "rgb":
        color_ctrl_seq = _color_ctrl_seq_24bits
    else:
        raise ValueError("The invalid color_scheme parameter.")

    fg_ctrl_seq = color_ctrl_seq(color=fg, fg=True)
    bg_ctrl_seq = color_ctrl_seq(color=bg, fg=False)
    ctrl_seq = fg_ctrl_seq + bg_ctrl_seq + bold_ctrl_seq
    return ctrl_seq + string + RESET
# }}}

def cprint(*objs, sep=' ', end='\n', file=None, flush=False,
        fg=None, bg=None, bold=False, color_scheme=8):# {{{
    """
    Print the given objs with color


    color_scheme is 4:
        'fg': ["black" | "red" | "green" | "yellow" | "blue" | "magenta" |
        "cyan" | "white" | 'k' | 'r' | 'g' | 'y' | 'b' | 'm' | 'c' | 'w' |
        30~37 | 90~97 | None]

        'bg': ["black" | "red" | "green" | "yellow" | "blue" | "magenta" |
        "cyan" | "white" | 'k' | 'r' | 'g' | 'y' | 'b' | 'm' | 'c' | 'w' |
        40~47 | 100~107 | None]

    color_scheme is 8:
        'fg' and 'bg': ["black" | "red" | "green" | "yellow" | "blue" |
        "magenta" | "cyan" | "white" | 'k' | 'r' | 'g' | 'y' | 'b' | 'm' |
        'c' | 'w' | 0~255 | None]

    color_scheme is 24 or "rgb":
        'fg' and 'bg': ["black" | "red" | "green" | "yellow" | "blue" |
        "magenta" | "cyan" | "white" | 'k' | 'r' | 'g' | 'y' | 'b' | 'm' |
        'c' | 'w' | (R, G, B) | [R, G, B] | "#XXXXXX" | None]

        R, G, B all in the range 0~255 which represent the red, green, blue
        components respectively.
        X is hex-digit, the first two Xs represent the red component, the
        third and fourth Xs represent the green component and the last two Xs
        represent the blue component.

    Note:
        This function is aimed at print string with color to the screen.
        It is not apporpriate to print to a text file although it is allowed by
        passing a file object to the file keyword argument.
        If printing to a text file, those ANSI escape sequence will appear
        as the content of the text file instead of changing the color of the
        string.

    Parameters
    ----------
    objs : A sequence of objects
    sep : str, optional, keyword-only
        string inserted between objs
        default : a space
    end : str, optional, keyword-only
        string appended after the last object
        default: a newline
    file : file like object (stream), optional, keyword-only
        default: the current sys.stdout
    flush : boolean, optional, keyword-only
        Whether to forcibly flush the stream
        default : False
    fg : None | str | int | tuple | list, optional, keyword-only
        The foreground color of the string
        default : None
    bg : None | str | int | tuple | list, optional, keyword-only
        The background color of the string
        default : None
    bold : boolean, optional, keyword-only
        Whether to use bold font
        default : False
    color_scheme : 4 | 8 | 24 | "rgb", keyword-only
        The color scheme to use
        "rgb" is the same as 24.
        default : 8

    See also
    --------
    https://en.wikipedia.org/wiki/ANSI_escape_code
    builtin print function
    """

    tmp = sep.join(str(obj) for obj in objs)
    colored_tmp = colored(tmp, fg=fg, bg=bg, bold=bold, color_scheme=color_scheme)
    print(colored_tmp, end=end, file=file, flush=flush)
# }}}

def cdemo(with_bg=False, space_num=0):# {{{
    """
    Demonstarte all colors of the 8-bits color scheme

    Parameter
    ---------
    with_bg : boolean, optional
        Whether to use background color or not
        default : False
    space_num : int, optional
        The number of space before the printed string.
        default: 0
    """

    prefix = ' ' * space_num
    print(prefix, end='')
    cprint("fg = None, bg = None", fg=None, bg=None)
    for fg in range(256):
        print(prefix, end='')
        cprint("fg = {0:>4d}, bg = None".format(fg), fg=fg, bg=None)

    if with_bg:
        for bg in range(256):
            print(prefix, end='')
            cprint("fg = None, bg = {0:>4d}".format(bg), fg=None, bg=bg)
        for fg in range(256):
            for bg in range(256):
                content = "fg = {0:>4d}, bg = {1:>4d}".format(fg, bg)
                print(prefix, end='')
                cprint(content, fg=fg, bg=bg)
# }}}

if __name__ == "__main__":
    from itertools import product

    colors_std_long = ["black", "red", "green", "yellow","blue", "magenta", "cyan", "white"]
    colors_std_short= ['k', 'r', 'g', 'y', 'b', 'm', 'c', 'w']
    #colors_codes_fg = list(range(30, 38)) + list(range(90, 98))
    #colors_codes_bg = list(range(40, 48)) + list(range(100, 108))
    colors_codes_fg = list(range(256))
    colors_codes_bg = list(range(256))
    colors_fg = colors_std_long + colors_std_short + colors_codes_fg + [None]
    colors_bg = colors_std_long + colors_std_short + colors_codes_bg + [None]
    for index, fg in enumerate(product(range(256), repeat=3)):
        cprint(" " * 60 + str(index), fg=fg, bold=True, color_scheme=24)
    #hex_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #        'a', 'b', 'c', 'd', 'e', 'f']
    #for index, cfg in enumerate(product(hex_digits, repeat=6)):
    #    cfg = "".join(cfg)
    #    fg = "".join(['#', cfg])
    #    cprint(" " * 60 + str(index), fg=fg, bold=True, color_scheme=24)
