
# import nidaqmx
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import chirp, fftconvolve
# from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# # === Parameters ===
# sample_rate = 100000  # Hz
# duration = 5  # seconds
# f_start = 20  # Hz
# f_end = 20000  # Hz
# amplitude = 0.5  # Volts

# # === Generate logarithmic sweep ===
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# sweep = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
# sweep_inverted = -sweep

# # Stack for balanced output (2 channels)
# output_data = np.vstack((sweep, sweep_inverted))

# # === DAQ Tasks ===
# with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
#     # Configure output (AO0 and AO1)
#     write_task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
#     write_task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)
#     write_task.timing.cfg_samp_clk_timing(
#         rate=sample_rate,
#         sample_mode=AcquisitionType.FINITE,
#         samps_per_chan=len(sweep))
    
#     # Configure input (AI0 - microphone)
#     read_task.ai_channels.add_ai_voltage_chan(
#         "Dev1/ai0",
#         min_val=-1.0,
#         max_val=1.0,
#         terminal_config=TerminalConfiguration.RSE)
#     read_task.timing.cfg_samp_clk_timing(
#         rate=sample_rate,
#         sample_mode=AcquisitionType.FINITE,
#         samps_per_chan=len(sweep))
    
#     # Execute measurement
#     print("Starting measurement...")
#     read_task.start()
#     write_task.write(output_data, auto_start=True)
#     recorded_data = np.array(read_task.read(number_of_samples_per_channel=len(sweep), timeout=duration+2))
#     write_task.wait_until_done()
#     read_task.stop()

# # === Impulse Response Calculation ===
# sweep_inverse = np.flip(sweep) / (np.max(sweep)**2)  # Inverse filter
# impulse_response = fftconvolve(recorded_data, sweep_inverse, mode='same')

# # Apply window to impulse response to reduce spectral leakage
# window = np.hanning(len(impulse_response))
# windowed_ir = impulse_response * window

# # Compute FFT
# n_fft = 2**18  # Zero-pad for better frequency resolution
# freq = np.fft.rfftfreq(n_fft, d=1/sample_rate)
# ir_fft = np.fft.rfft(windowed_ir, n=n_fft)

# # Convert to dB scale (avoid log of zero)
# magnitude_db = 20 * np.log10(np.abs(ir_fft) + 1e-10)  # Add small offset to avoid log(0)

# # === Enhanced Plotting ===
# plt.figure(figsize=(12, 12))

# # 1. Time Domain: Impulse Response
# plt.subplot(3, 1, 1)
# plt.plot(t, impulse_response)
# plt.title('Impulse Response (Time Domain)')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.xlim([0, 0.05])  # Zoom in on first 50ms

# # 2. Frequency Domain: Magnitude Response
# plt.subplot(3, 1, 2)
# plt.semilogx(freq, magnitude_db)
# plt.title('Frequency Response (Magnitude)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.grid(True, which='both')
# plt.xlim([20, 20000])  # Show full audio range


# # # === Plotting ===
# # plt.figure(figsize=(12, 10))

# # # 3. Impulse Response
# # plt.subplot(3, 1, 3)
# # plt.plot(t, impulse_response, label='Impulse Response')
# # plt.title('Extracted Impulse Response')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')
# # plt.grid(True)

# # plt.tight_layout()
# # plt.show()




import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, fftconvolve
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# === Parameters ===
sample_rate = 96_000  # Hz
duration = 1  # seconds
f_start = 20  # Hz
f_end = 20000  # Hz
amplitude = 0.5  # Volts

# === Generate logarithmic sweep ===
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sweep = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
sweep_inverted = -sweep

# Stack for balanced output (2 channels)
output_data = np.vstack((sweep, sweep_inverted))

# === DAQ Tasks ===
with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
    # Configure output (AO0 and AO1)
    write_task.ao_channels.add_ao_voltage_chan(
        "Dev1/ao0",
        min_val=-10.0,
        max_val=10.0
        )
    write_task.ao_channels.add_ao_voltage_chan(
        "Dev1/ao1",
        min_val=-10.0,
        max_val=10.0
        )
    write_task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=len(sweep)
        )
    
    # Configure input (AI0 - microphone)
    read_task.ai_channels.add_ai_voltage_chan(
        "Dev1/ai0",
        min_val=-1.0,
        max_val=1.0,
        terminal_config=TerminalConfiguration.RSE)
    read_task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=len(sweep)
        )
    
    # Execute measurement
    print("Starting measurement...")
    read_task.start()
    write_task.write(output_data, auto_start=True)
    recorded_data = np.array(read_task.read(number_of_samples_per_channel=len(sweep), timeout=duration+2))
    write_task.wait_until_done()
    read_task.stop()


# === Impulse Response Calculation ===
sweep_inverse = np.flip(sweep) / (np.max(sweep)**2)  # Inverse filter
impulse_response = fftconvolve(recorded_data, sweep_inverse, mode='same')


# Apply window to impulse response to reduce spectral leakage
window = np.hanning(len(impulse_response))
windowed_ir = impulse_response * window


# Compute FFT
n_fft = 2**18  # Zero-pad for better frequency resolution
freq = np.fft.rfftfreq(n_fft, d=1/sample_rate)
ir_fft = np.fft.rfft(windowed_ir, n=n_fft)

# Convert to dB scale (avoid log of zero)
magnitude_db = 20 * np.log10(np.abs(ir_fft) + 1e-10)  # Add small offset to avoid log(0)


# === Plotting ===
plt.figure(figsize=(12, 10))

# 2. Frequency Domain: Magnitude Response
plt.subplot(3, 1, 2)
plt.semilogx(freq, magnitude_db)
plt.title('Frequency Response (Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True, which='both')
plt.xlim([20, 20000])  # Show full audio range


# 3. Impulse Response
plt.subplot(3, 1, 3)
plt.plot(t, impulse_response, label='Impulse Response')
plt.title('Extracted Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()