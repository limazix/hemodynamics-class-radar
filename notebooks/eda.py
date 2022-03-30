#%%
import os

import pandas as pd
import numpy as np

import scipy.io as sio
from scipy import signal
from scipy.fft import fft, ifft, fftfreq

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
# %%
DATA_ROOT_PATH=os.path.abspath('../data')

SUBJECT_ID_PREFIX='GDN00'
SUBJECT_IDS=[SUBJECT_ID_PREFIX + str(i) if i > 9 else '{}0{}'.format(SUBJECT_ID_PREFIX, i) for i in range(1, 31)]

SCENARIOS=['Resting', 'Valsalva', 'Apnea', 'TiltUp', 'TiltDown']
# %%
# Evaluating a Subject's Data
def load_data(filepath):
    return sio.loadmat(filepath)

def mat_to_dict(mat):
    return {k: np.array(v).flatten() for k, v in mat.items() if k[0] != '_'}
# %%
SUBJECT_ID=SUBJECT_IDS[3]
SCENARIO=SCENARIOS[0]

RESTING_PATH=os.path.join(DATA_ROOT_PATH, SUBJECT_ID, '{}_1_{}.mat'.format(SUBJECT_ID, SCENARIO))
# %%
mat = load_data(RESTING_PATH)
mat.keys()
# %%
data = mat_to_dict(mat)

# %%

data['radar_i'].shape, data['radar_q'].shape, data['fs_radar'].shape
# %%
radar_sample_rate = data['fs_radar'][0]
bp_sample_rate = data['fs_bp'][0]
# %%
radar_time_btw_sample = 1/radar_sample_rate
bp_time_btw_sample = 1/bp_sample_rate
# %%
radar_num_samples = len(data['radar_i'])
bp_num_samples = len(data['tfm_bp'])
# %%
radar_sample_time = [radar_time_btw_sample*i for i in range(radar_num_samples)]
bp_sample_time = [bp_time_btw_sample*i for i in range(bp_num_samples)]

bp_sample_rate
# %%

# 1s * n
radar_timeframe=radar_sample_rate * 10
bp_timeframe=bp_sample_rate * 10

fig = go.Figure()

fig.add_trace(go.Line(x=radar_sample_time[:radar_timeframe], y=data['radar_i'][:radar_timeframe]))
fig.add_trace(go.Line(x=radar_sample_time[:radar_timeframe], y=data['radar_q'][:radar_timeframe]))
fig.add_trace(go.Line(x=bp_sample_time[:bp_timeframe], y=data['tfm_bp'][:bp_timeframe]))

fig.show()
# %%
# Downsample
rradar_q = signal.resample(data['radar_q'], bp_num_samples)
rradar_i = signal.resample(data['radar_i'], bp_num_samples)

# %%
fig = go.Figure()

fig.add_trace(go.Line(x=bp_sample_time[:bp_timeframe], y=rradar_i[:bp_timeframe]))
fig.add_trace(go.Line(x=bp_sample_time[:bp_timeframe], y=rradar_q[:bp_timeframe]))

fig.show()

# %%
# Butterworth Filtering

# Nyquist Frequency
nyq = bp_sample_rate*0.5

b, a = signal.butter(4, [0.01/nyq, 20.0/nyq], 'bandpass')
zi = signal.lfilter_zi(b, a)

frradar_q, _ = signal.lfilter(b, a, rradar_q, zi=zi*rradar_q[0])
frradar_i, _ = signal.lfilter(b, a, rradar_i, zi=zi*rradar_i[0])

# %%
fig = go.Figure()

fig.add_trace(go.Line(x=bp_sample_time[:bp_timeframe], y=frradar_i[:bp_timeframe]))
fig.add_trace(go.Line(x=bp_sample_time[:bp_timeframe], y=frradar_q[:bp_timeframe]))

fig.show()

# %%
# Arctangent Demodulation

raw_distance = np.arctan(rradar_q/rradar_i)

# %%
fig = go.Figure()

fig.add_trace(go.Line(y=raw_distance[:bp_timeframe], x=bp_sample_time[:bp_timeframe]))
fig.add_trace(go.Line(y=data['tfm_bp'][:bp_timeframe], x=bp_sample_time[:bp_timeframe]))

fig.show()
# %%
