import nidaqmx
import numpy as np
import time
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter

# Parameters
sample_rate = 96_000  # Hz
duration = 0.5  # seconds
amplitude = 0.5  # Volts (peak amplitude)
num_samples = int(sample_rate * duration)

# Generate white noise (Gaussian distribution)
np.random.seed(0)  # For reproducibility
white_noise = amplitude * np.random.normal(0, 1, num_samples)  # Mean=0, Std=1

# Stack into 2D array (2 channels x N samples)
data = np.vstack((white_noise, -white_noise))

with nidaqmx.Task() as task:
    # Add AO channels
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)

    # Configure sample clock for continuous generation
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS
    )

    # Use stream writer for multi-channel output
    writer = AnalogMultiChannelWriter(task.out_stream, auto_start=True)
    writer.write_many_sample(data)  # Shape: (2 channels, num_samples)

    print("Generating balanced white noise... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)  # Reduce CPU usage
    except KeyboardInterrupt:
        print("Stopped.")