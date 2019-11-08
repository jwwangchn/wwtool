import PIL
import numpy as np
import wwtool

def get_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0, n):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while (lab > 0):
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return palette

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for _ in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')
    return new_mask

def show_mask(mask, num_classes):
    palette = get_palette(num_classes)
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask_np = np.array(colorized_mask)
    colorized_mask_np = colorized_mask_np[:, :, ::-1].copy()
    wwtool.show_image(colorized_mask_np, wait_time=500)
    