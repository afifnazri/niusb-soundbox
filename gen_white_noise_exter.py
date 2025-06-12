import nidaqmx
import numpy as np
import time
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter

# Parameters
sample_rate = 100_000  # Hz
duration = 0.5  # seconds
frequency = 1000  # Hz
# amplitude_sine = 0.2  # same as noise
amplitude_sine = 0.3 # 6db higer than noise
amplitude_noise = 0.2  # Volts (noise amplitude)
num_samples = int(sample_rate * duration)

# Generate time array
t = np.linspace(0, duration, num_samples, endpoint=False)

# Generate signals
sine_wave = amplitude_sine * np.sin(2 * np.pi * frequency * t)
white_noise = amplitude_noise * np.random.normal(0, 1, num_samples)

# combined_signal = white_noise
# combined_signal = sine_wave
combined_signal = white_noise + sine_wave
data = np.vstack((combined_signal, -combined_signal))

with nidaqmx.Task() as task:
    # Add AO channels
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)

    # Configure sample clock
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS
    )

    # Use stream writer
    writer = AnalogMultiChannelWriter(task.out_stream, auto_start=True)
    writer.write_many_sample(data)

    print("Generating balanced (noise + 1kHz) on AO0 and inverted on AO1...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped.")