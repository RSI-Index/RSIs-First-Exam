import colorsys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase

font_dir = ["FONTS"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

plt.rcParams.update(
    {
        "font.family": "Manrope",
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.labelcolor": "#333333",
        "legend.fontsize": 12,
        "text.color": "#333333",
        "font.size": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


class GradientHandler(HandlerBase):
    def __init__(self, cmap, num_stripes=20, adjust_stripe_ycenter=0.0, **kw):
        super().__init__(**kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.adjust_stripe_ycenter = adjust_stripe_ycenter

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # stripe_height = self.stripe_height_ratio * fontsize
        y_center = ydescent + fontsize * self.adjust_stripe_ycenter

        stripes = []
        for i in range(self.num_stripes):
            color = self.cmap(i / (self.num_stripes - 1))
            s = mpatches.Rectangle(
                (xdescent + width * i / self.num_stripes, y_center),
                width / self.num_stripes,
                height,
                color=color,
                transform=trans,
                linewidth=0,
            )
            stripes.append(s)
        return stripes


def hex_to_hsl(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (
        int(hex_color[:2], 16) / 255,
        int(hex_color[2:4], 16) / 255,
        int(hex_color[4:], 16) / 255,
    )
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def change_lightness(hex_color, factor=1.0, delta=0.0):
    assert factor == 1.0 or delta == 0.0
    h, s, l = hex_to_hsl(hex_color)
    l = min(1, max(0, l * factor + delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
