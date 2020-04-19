import random
from mmcv.utils import is_str

DOTA_CLASS_NAMES = ['__background__', 'harbor', 'ship', 'small-vehicle', 'large-vehicle', 'storage-tank', 'plane', 'soccer-ball-field', 'bridge', 'baseball-diamond', 'tennis-court', 'helicopter', 'roundabout', 'swimming-pool', 'ground-track-field', 'basketball-court']

DOTA_CLASS_NAMES_OFFICIAL = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

# (R, G, B)
COLORS = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

# # mask obb (v301)
# COLORS = {'Green': (60, 180, 75), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Blue': (0, 130, 200), 
#         'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 
#         'Lavender': (230, 190, 255), 'Teal': (0, 128, 128), 'Lime': (210, 245, 60), 'Pink': (250, 190, 190), 
#         'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 
#         'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 
#         'White': (255, 255, 255), 'Black': (0, 0, 0)}

# v302, v309, v310
# COLORS = {'Red': (230, 25, 75), 'Green': (60, 180, 75), 'Yellow': (255, 225, 25), 'Blue': (0, 130, 200), 
#         'Pink': (250, 190, 190), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Maroon': (128, 0, 0), 
#         'Magenta': (240, 50, 230), 'Cyan': (70, 240, 240), 'Beige': (255, 250, 200), 'Lavender': (230, 190, 255), 
#         'Teal': (0, 128, 128), 'Brown': (170, 110, 40), 'Lime': (210, 245, 60)}

# v301, v302, v309, v310 公用
# COLORS = {'Red': (230, 25, 75), 'Green': (60, 180, 75), 'Yellow': (255, 225, 25), 'Blue': (0, 130, 200), 'Pink': (250, 190, 190), 'Orange': (245, 130, 48), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Cyan': (70, 240, 240), 'Lavender': (240, 50, 230), 'Teal': (70, 240, 240), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Brown': (170, 110, 40), 'Lime': (210, 245, 60)}

# DOTA dataset
# COLORS = {'harbor': (60, 180, 75), 'ship': (230, 25, 75), 'small-vehicle': (255, 225, 25), 'large-vehicle': (245, 130, 200), 
#         'storage-tank': (230, 190, 255), 'plane': (245, 130, 48), 'soccer-ball-field': (0, 0, 128), 'bridge': (255, 250, 200), 
#         'baseball-diamond': (240, 50, 230), 'tennis-court': (70, 240, 240), 'helicopter': (0, 130, 200), 'roundabout': (170, 255, 195), 
#         'swimming-pool': (250, 190, 190), 'ground-track-field': (170, 110, 40), 'basketball-court': (0, 128, 128)}

# small
# COLORS = {'0': (230, 25, 75), '1': (60, 180, 75), '2': (255, 225, 25), '3': (0, 130, 200), '4': (250, 190, 190), '5': (245, 130, 48), '6': (70, 240, 240), '7': (240, 50, 230), '8': (230, 190, 255)}

def color_val(color = None):
    # COLORS = {k: v for k, v in sorted(COLORS.items(), key=lambda item: item[0])}
    if is_str(color):
        color = color[0].upper() + color[1:].lower()
        return list(COLORS[color])[::-1]
    elif color == None:
        color_name = random.choice(list(COLORS.keys()))
        return list(COLORS[color_name])[::-1]
    elif type(color) == int:
        return list(COLORS[list(COLORS.keys())[color]])[::-1]
    else:
        return list(COLORS['Red'])[::-1]