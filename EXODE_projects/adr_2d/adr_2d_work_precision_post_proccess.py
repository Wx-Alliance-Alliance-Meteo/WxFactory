import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import to_rgb, to_hex
import colorsys


def read_data(filename):
    time = []
    error = [] 
    with open(filename, 'r') as f:
        array=np.loadtxt(filename) 
        time = array[:,5]
        error = array[:, 3]
    return time, error    

def get_min_time(method,epi,tol):
    time = []
    TIME = [] # collect all time data
    min_time = []
    error = []
    for repeat in ["0","1","2","3","4"]: 
        data_dir = "/home/siw001/hall5/testoutput/ADR_2D/"+epi+"/rerun_full_node_rtol_"+rtol+"/"
        filename = method+"_tol_"+tol+"_rtol_"+rtol+"_Nx_"+Nx+"_nt_50_400_repeat_"+repeat + ".txt"
        time,error = read_data(data_dir+filename)
        TIME.append(time) 
    min_time = np.min(TIME,axis=0)
    return min_time, error


tol = "1e-10"
rtol = "1e-3"
Nx = "401"
epi = "epi5"

line_width = 3
label_fontsize = 16
legend_fontsize = 14
tick_fontsize = 14
marker_size = 8

ALL_METHODS = ["pmex",
        "kiops",
        "BS32",
        "KC32",
        "M43",
        "DP54",
        "EXLRK32",
        #"EXLRK43",
        "ExLRK4(3)minA5",
        ]

PAPER_NAME = [
        "PMEX",
        "KIOPS",
        "BS3(2)",
        "KC3(2)",
        "M4(3)",
        "DP5(4)",
        "ExLRK3(2)",
        "ExLRK4(3)",
        #"ExLRK4(3)minA5",
        ]

# === Marker classification
MARKER_GROUPS =  {
    'o': ['pmex','kiops'],          # Krylov
    's': ['BS32', 'KC32', 'EXLRK32'], # 3rd order
    'x': ['M43', 'EXLRK43', 'ExLRK4(3)minA5'],  # 4th order
    '+': ['DP54']   # 5th order
}




# === Linestyle classification
LINESTYLE_GROUPS = {
    '--': ['pmex', 'kiops'],  # Krylov
    '-.': ['BS32', 'KC32', 'M43','DP54'],  # Classical
    '-':  ['EXLRK32', 'EXLRK43','ExLRK4(3)minA5']  # EXODE
}



# Function to generate shades of a base color
def generate_shades(base_color, n_shades,lightness_min=0.25, lightness_max=0.6):
    """Generate n_shades variations of base_color by changing lightness."""
    rgb = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    shades = []

    for i in range(n_shades):
        # Linearly vary lightness between 0.4 and 0.8 for visible shades
        new_l = lightness_min + (lightness_max - lightness_min) * i / (n_shades - 1)
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        shades.append(new_rgb)
    return shades


# === Base color mapping (line groups will share base color) ===
BASE_COLORS = {
    '--': '#4C72B0',   # soft blue for Krylov
    '-.': '#DD8452',   # muted orange for Classic
    '-': '#2ca02c',   # green for EXODE
}


LIGHTNESS_RANGES = {
    '--': (0.30, 0.50),  # Blue
    '-.': (0.30, 0.55),  # Orange
    '-': (0.25, 0.60)   # Green
}

# === Generate shades for each marker group ===
color_map = {}
method_marker = {}
method_linestyle = {}

for linestyle, methods in LINESTYLE_GROUPS.items():
    base_color = BASE_COLORS[linestyle]
    lmin, lmax = LIGHTNESS_RANGES[linestyle]
    shades = generate_shades(base_color, len(methods), lightness_min=lmin, lightness_max=lmax)
    for method, shade in zip(methods, shades):
        color_map[method] = shade
        method_linestyle[method] = linestyle

for mk, methods in MARKER_GROUPS.items():
    for method in methods:
        method_marker[method] = mk

plt.figure(figsize=(12, 6))


for method in ALL_METHODS:
    marker_style = method_marker[method]
    line_style = method_linestyle[method]
    shade = color_map[method]

    time, error=get_min_time(method, epi, tol)
    plt.semilogy(time, error,  linewidth=line_width, color=shade, linestyle = line_style, marker=marker_style, markersize = marker_size, markeredgewidth=2, label=PAPER_NAME[ALL_METHODS.index(method)])



plt.xlabel('Time (s)',fontsize = label_fontsize)
plt.ylabel('Error',fontsize = label_fontsize)
plt.xticks(fontsize = tick_fontsize)
plt.yticks(fontsize = tick_fontsize)
#plt.title("Work precision diagram for ADR with "+epi+" tol " + tol)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(loc='upper right', fontsize=legend_fontsize) #,bbox_to_anchor=(1.05, 1))


plt.tight_layout()
plt.savefig("./testoutput_fig/ADR_work_precision_"+ epi +"_tol_"+tol+"_rtol_"+rtol+"_nt_50_400.png", dpi = 300)
 
print("fig saved to " + "./testoutput_fig/ADR_work_precision_"+ epi +"_tol_"+tol+"_rtol_"+rtol+"_nt_50_400.png")
