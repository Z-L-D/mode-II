import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Paste the content of gap_a.txt into a string variable or read from file.
# For brevity, I'll simulate the extraction using the data I manually collected from the logs.

# I'll create arrays for each run (step, g). I'll use the numbers I saw in the logs.
# For Gen 0 (512x512, 1120 tokens) – from the log, g values at steps: 0:0.0196, 1:0.0195, 2:0.0194, 3:0.0191, 4:0.0189, 5:0.0184, 6:0.0177, 7:0.0171, 8:0.0167, 9:0.0165, 10:0.0162, 11:0.0159, 12:0.0156, 13:0.0154, 14:0.0153, 15:0.0153, 16:0.0153, 17:0.0155, 18:0.0157, 19:0.0159, 20:0.0161, ... up to 99.
# This is too much to type manually. Instead, I'll use the averaged fixed-point values and the fact that the shape is similar across runs, but that's not enough for β(g) functional form.

# Given the difficulty of manual extraction, I'll assume you have the data in a structured format.
# I'll provide the analysis code that you can run on your parsed data.

def parse_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    runs = []
    current_run = []
    for line in lines:
        if 'g_micro=' in line:
            match = re.search(r'g_micro=([0-9.]+)', line)
            if match:
                g = float(match.group(1))
                current_run.append(g)
        elif 'Requested to load AutoencodingEngine' in line:
            if current_run:
                runs.append(np.array(current_run))
                current_run = []
    return runs

# After parsing, you'll have a list of 5 arrays, each of length 100 (g per step).
# Then compute β per step: beta[t] = g[t+1] - g[t] for t=0..98.
# Average across runs.

runs = parse_log('gap_a.txt')  # replace with your file
g_steps = np.array(runs)  # shape (5, 100)
beta = np.diff(g_steps, axis=1)  # shape (5, 99)

# Average beta across runs
beta_mean = np.mean(beta, axis=0)
g_mean = np.mean(g_steps[:, :-1], axis=0)  # align steps

# Now fit polynomial to beta_mean as function of g_mean
def parabola(g, a, b, c):
    return a*g**2 + b*g + c

def cubic(g, a, b, c, d):
    return a*g**3 + b*g**2 + c*g + d

popt_par, _ = curve_fit(parabola, g_mean, beta_mean)
popt_cub, _ = curve_fit(cubic, g_mean, beta_mean)

print("Parabola coefficients:", popt_par)
print("Cubic coefficients:", popt_cub)