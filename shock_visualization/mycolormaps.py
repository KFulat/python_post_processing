from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

colors = ["#03071e", "#053c5e", "#51537B", "#D8313C", "#fcbf49", "#fdffb6"]
cmap_modern = LinearSegmentedColormap.from_list("modern", colors)

a = -2*224
top = cm.get_cmap('jet', 2*256+a)
bottom = cm.get_cmap('gray', 2*256-a)
newcolors = np.vstack((top(np.linspace(0, 1, 2*256+a)),
                        bottom(np.linspace(0, 1, 2*256-a))))
cmap_turb = ListedColormap(newcolors)
style_color = [[0, 0, 127],
               [0, 0, 197],
               [0, 21, 254],
               [0, 126, 254],
               [0, 231, 254],
               [68, 253, 186],
               [153, 254, 101],
               [238, 254, 16],
               [254, 187, 0],
               [254, 101, 0],
               [254, 16, 0],
               [197, 0, 0],
               [127, 0, 0],
               [127, 0, 0]]

# transform color rgb value to 0-1 range
color_arr = []
for color in style_color:
    rgb = [float(value) / 255 for value in color]
    color_arr.append(rgb)
colors_undersea = cm.jet(np.linspace(0, 1, 256))
# colors_land = cm.terrain(np.linspace(0.25, 1, 256))
colors_land = cm.gray(np.linspace(0, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
cmap_turb = LinearSegmentedColormap.from_list(
    'terrain_map', all_colors)
# cmap_turb = LinearSegmentedColormap.from_list('my_palette', color_arr, N=256)
# a = 2*250
# middle = cm.get_cmap('jet', 2*256-a)
# top = cm.get_cmap('gray_r', 2*256+a)
# bottom = cm.get_cmap('gray', 2*256+a)
# newcolors = np.vstack((top(np.linspace(0, 1, 2*256+a)),
#                     middle(np.linspace(0, 1, 2*256-a)),
#                         bottom(np.linspace(0, 1, 2*256+a))))
# cmap_turb = ListedColormap(newcolors)
magma = cm.get_cmap('magma', 256)
cmap_ft = magma(np.linspace(0, 1, 256))
white = np.array([256/256, 256/256, 256/256, 1])
cmap_ft[:10] = white
cmap_ft = ListedColormap(cmap_ft)