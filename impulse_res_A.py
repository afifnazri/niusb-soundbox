import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, fftconvolve, bilinear, lfilter
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# === Parameters ===
sample_rate = 96_000  # Hz
samples_per_read = 256_000  # Now the primary control parameter
duration = samples_per_read / sample_rate  # Calculated duration in seconds
f_start = 20  # Hz
f_end = 20000  # Hz
amplitude = 0.5  # Volts
# duration = 1  # seconds

# === Generate logarithmic sweep ===
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sweep = amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
sweep_inverted = -sweep

# Stack for balanced output (2 channels)
output_data = np.vstack((sweep, sweep_inverted))

def a_weighting(fs):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * np.pi * f4)**2 * (10**(A1000 / 20)),
            0, 0, 0, 0]
    DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
                       [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
    DENs = np.convolve(np.convolve(DENs,[1, 2 * np.pi * f3]),
                       [1, 2 * np.pi * f2])

    b, a = bilinear(NUMs, DENs, fs)
    return b, a
b, a = a_weighting(sample_rate)


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

# === Apply A-weighting to recorded sweep BEFORE deconvolution ===
weighted_recorded_data = lfilter(b, a, recorded_data)



# === Impulse Response Calculation ===
sweep_inverse = np.flip(sweep) * (10**(-6/20))  # -6 dB gain compensation
impulse_response = fftconvolve(weighted_recorded_data, sweep_inverse, mode='same')

# Apply window to impulse response to reduce spectral leakage
window = np.hanning(len(impulse_response))
windowed_ir = impulse_response * window


# Compute FFT
n_fft = 2**18
freq = np.fft.rfftfreq(n_fft, d=1/sample_rate)
ir_fft = np.fft.rfft(windowed_ir, n=n_fft)
magnitude_db = 20 * np.log10(np.abs(ir_fft) / np.mean(window) + 1e-10)

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