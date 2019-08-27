import random
from mmcv.utils import is_str

# (R, G, B)
COLORS = {'Red': (230, 25, 75), 'Green': (60, 180, 75), 'Yellow': (255, 225, 25), 'Blue': (0, 130, 200), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lime': (210, 245, 60), 'Pink': (250, 190, 190), 'Teal': (0, 128, 128), 'Lavender': (230, 190, 255), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

def color_val(color = None):
    if is_str(color):
        color = color[0].upper() + color[1:].lower()
        return list(COLORS[color])[::-1]
    elif color == None:
        color_name = random.choice(list(COLORS.keys()))
        return list(COLORS[color_name])[::-1]
    else:
        return list(COLORS['Red'])[::-1]