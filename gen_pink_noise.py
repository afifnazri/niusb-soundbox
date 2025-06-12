import nidaqmx
import numpy as np
import time
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from scipy.signal import lfilter


# Parameters
sample_rate = 96_000  # Hz
duration = 0.5  # seconds
amplitude = 1  # Volts (peak amplitude)
num_samples = int(sample_rate * duration)

# Pink noise generation function
def generate_pink_noise(N):
    # This implements the Voss-McCartney algorithm for pink noise
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    white = np.random.normal(0, 1, N + 48)  # Extra samples for filter delay
    pink = lfilter(b, a, white)[48:]  # Remove initial transient
    return pink / np.max(np.abs(pink))  # Normalize

# Generate pink noise
pink_noise = amplitude * generate_pink_noise(num_samples)
inverted_pink = -pink_noise  # Balanced output

# Stack into 2D array (2 channels x N samples)
data = np.vstack((pink_noise, inverted_pink))

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

    print("Generating balanced pink noise... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)  # Reduce CPU usage
    except KeyboardInterrupt:
        print("Stopped.")