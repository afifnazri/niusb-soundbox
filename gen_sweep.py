import nidaqmx
import numpy as np
import time
from scipy.signal import chirp
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter

# === Parameters ===
sample_rate = 100_000  # High enough to cover 20 kHz (min 2x per Nyquist)
duration = 5  # seconds
f_start = 20    # Hz
f_end = 20000   # Hz
amplitude = 0.5  # Volts (safe within ±10 V range of USB-6212)

# === Generate sweep (chirp) signal ===
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sweep = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
sweep_inverted = -sweep

data = np.vstack((sweep, sweep_inverted))

# === Output sweep signal ===
with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)

    # Configure sample clock
    task.timing.cfg_samp_clk_timing(rate=sample_rate,
                                    sample_mode=AcquisitionType.CONTINUOUS)

    # Use stream writer for multi-channel write
    writer = AnalogMultiChannelWriter(task.out_stream, auto_start=True)
    writer.write_many_sample(data)  # data shape: (num_channels, num_samples) 

    print(f"Playing sweep tone from {f_start} Hz to {f_end} Hz for {duration} seconds...")
    task.wait_until_done(timeout=duration + 1)
