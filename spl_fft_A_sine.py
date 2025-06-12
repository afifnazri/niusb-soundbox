# import nidaqmx
# from nidaqmx.constants import AcquisitionType, TerminalConfiguration
# from nidaqmx.stream_writers import AnalogMultiChannelWriter
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from scipy.signal import bilinear, lfilter

# # DAQ Settings
# sample_rate = 96_000
# # samples_per_read = 64 * 1024  # Buffer size for reading
# samples_per_read = 128 * 1024  # Buffer size for reading
# amplitude = 0.5  # Volts
# frequency = 1000  # Hz

# # Mic calibration
# ref_Pa = 20e-6
# mic_sensitivity = 14e-3  # V/Pa
# preamp_gain = 21  # dB

# # Generate output signal (balanced differential)
# t = np.linspace(0, samples_per_read / sample_rate, samples_per_read, endpoint=False)
# waveform = amplitude * np.sin(2 * np.pi * frequency * t)
# data = np.vstack((waveform, -waveform))  # 2 channels: ao0 and ao1

# # A-weighting filter
# def a_weighting(fs):
#     f1 = 20.598997
#     f2 = 107.65265
#     f3 = 737.86223
#     f4 = 12194.217
#     A1000 = 1.9997
#     NUMs = [(2 * np.pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0]
#     DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
#                       [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
#     DENs = np.convolve(np.convolve(DENs, [1, 2 * np.pi * f3]),
#                        [1, 2 * np.pi * f2])
#     b, a = bilinear(NUMs, DENs, fs)
#     return b, a

# b, a = a_weighting(sample_rate)

# # Initialize plot
# plt.ion()
# fig, ax = plt.subplots()
# x = np.fft.rfftfreq(samples_per_read, 1/sample_rate)
# line, = ax.semilogx(x, np.zeros_like(x))
# ax.set_xlim(20, 20000)
# ax.set_ylim(-200, 0)
# ax.set_xlabel('Frequency (Hz)')
# ax.set_ylabel('Magnitude (dB)')
# ax.grid(True, which='both', ls='--')

# # Create separate tasks
# output_task = nidaqmx.Task()
# input_task = nidaqmx.Task()

# # Configure output task (AO)
# output_task.ao_channels.add_ao_voltage_chan(
#     "Dev1/ao0",
#     min_val=-10.0,
#     max_val=10.0,
#     )
# output_task.ao_channels.add_ao_voltage_chan(
#     "Dev1/ao1",
#     min_val=-10.0,
#     max_val=10.0
#     )
# output_task.timing.cfg_samp_clk_timing(
#     rate=sample_rate,
#     sample_mode=AcquisitionType.CONTINUOUS,
#     samps_per_chan=samples_per_read
# )

# # Configure input task (AI)
# input_task.ai_channels.add_ai_voltage_chan(
#     "Dev1/ai0",
#     terminal_config=TerminalConfiguration.DIFF,
#     min_val=-1.0,
#     max_val=1.0
# )
# input_task.timing.cfg_samp_clk_timing(
#     rate=sample_rate,
#     sample_mode=AcquisitionType.CONTINUOUS,  # ✅ Fixed parameter
#     samps_per_chan=samples_per_read
# )

# # Start output (cyclic regeneration)
# writer = AnalogMultiChannelWriter(output_task.out_stream, auto_start=True)
# writer.write_many_sample(data)  # Initial buffer

# # Start input
# input_task.start()

# try:
#     while True:
#         # Read input
#         raw_data = input_task.read(number_of_samples_per_channel=samples_per_read)
#         np_data = np.array(raw_data)

#         # Apply window before A-weighting
#         windowed = np_data * np.hanning(len(np_data))
#         weighted_data = lfilter(b, a, windowed)

#         # Compute RMS & SPL
#         rms_voltage = np.sqrt(np.mean(weighted_data**2))
#         voltage_mic = rms_voltage / (10**(preamp_gain / 20))
#         mic_Pa = voltage_mic / mic_sensitivity
#         spl_a = 20 * np.log10(mic_Pa / ref_Pa)

#         # FFT
#         fft_vals = np.fft.rfft(windowed)
#         mag = np.abs(fft_vals) / (samples_per_read / 2)  # Normalize
#         mag_db = 20 * np.log10(mag + 1e-12)

#         # Update plot
#         line.set_ydata(mag_db)
#         fig.canvas.flush_events()

#         print(f"SPL (A-weighted): {spl_a:.2f} dBSPLA")

# except KeyboardInterrupt:
#     print("Stopping...")
# finally:
#     input_task.stop()
#     output_task.stop()
#     input_task.close()
#     output_task.close()
#     plt.close()


import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import bilinear, lfilter

class AudioAnalyzer:
    def __init__(self, sample_rate=96000, samples_per_read=128*1024, amplitude=0.5, frequency=1000):
        # DAQ Settings
        self.sample_rate = sample_rate
        self.samples_per_read = samples_per_read
        self.amplitude = amplitude
        self.frequency = frequency
        
        # Mic calibration
        self.ref_Pa = 20e-6
        self.mic_sensitivity = 14e-3  # V/Pa
        self.preamp_gain = 21  # dB
        
        # Initialize tasks and streams
        self.output_task = None
        self.input_task = None
        self.writer = None
        
        # Initialize plot
        self.fig, self.ax = plt.subplots()
        self.x = np.fft.rfftfreq(self.samples_per_read, 1/self.sample_rate)
        self.line, = self.ax.semilogx(self.x, np.zeros_like(self.x))
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-200, 0)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.grid(True, which='both', ls='--')
        
        # Generate A-weighting filter coefficients
        self.b, self.a = self._create_a_weighting_filter()
        
        # Generate output waveform
        self.waveform = self._generate_waveform()
        
    def _create_a_weighting_filter(self):
        """Create A-weighting filter coefficients for the given sample rate"""
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
        NUMs = [(2 * np.pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0]
        DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
                          [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
        DENs = np.convolve(np.convolve(DENs, [1, 2 * np.pi * f3]),
                         [1, 2 * np.pi * f2])
        b, a = bilinear(NUMs, DENs, self.sample_rate)
        return b, a
    
    def _generate_waveform(self):
        """Generate the output waveform"""
        t = np.linspace(0, self.samples_per_read / self.sample_rate, 
                        self.samples_per_read, endpoint=False)
        waveform = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        return np.vstack((waveform, -waveform))  # 2 channels: ao0 and ao1
    
    def setup_output(self):
        """Configure and start the output task"""
        self.output_task = nidaqmx.Task()
        
        # Configure output channels
        self.output_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao0",
            min_val=-10.0,
            max_val=10.0,
        )
        self.output_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao1",
            min_val=-10.0,
            max_val=10.0
        )
        
        # Configure timing
        self.output_task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.samples_per_read
        )
        
        # Start output (cyclic regeneration)
        self.writer = AnalogMultiChannelWriter(self.output_task.out_stream, auto_start=True)
        self.writer.write_many_sample(self.waveform)  # Initial buffer
    
    def setup_input(self):
        """Configure and start the input task"""
        self.input_task = nidaqmx.Task()
        
        # Configure input channel
        self.input_task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0",
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-1.0,
            max_val=1.0
        )
        
        # Configure timing
        self.input_task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.samples_per_read
        )
        
        # Start input
        self.input_task.start()
    
    def read_spl(self):
        """Read input data and calculate SPL"""
        # Read input
        raw_data = self.input_task.read(number_of_samples_per_channel=self.samples_per_read)
        np_data = np.array(raw_data)
        
        # Apply window before A-weighting
        windowed = np_data * np.hanning(len(np_data))
        weighted_data = lfilter(self.b, self.a, windowed)
        
        # Compute RMS & SPL
        rms_voltage = np.sqrt(np.mean(weighted_data**2))
        voltage_mic = rms_voltage / (10**(self.preamp_gain / 20))
        mic_Pa = voltage_mic / self.mic_sensitivity
        spl_a = 20 * np.log10(mic_Pa / self.ref_Pa)
        
        return spl_a, windowed
    
    def update_plot(self, windowed_data):
        """Update the FFT plot with new data"""
        fft_vals = np.fft.rfft(windowed_data)
        mag = np.abs(fft_vals) / (self.samples_per_read / 2)  # Normalize
        mag_db = 20 * np.log10(mag + 1e-12)
        
        # Update plot
        self.line.set_ydata(mag_db)
        self.fig.canvas.flush_events()
    
    def run_measurement(self):
        """Run the continuous measurement loop"""
        try:
            while True:
                spl_a, windowed_data = self.read_spl()
                self.update_plot(windowed_data)
                print(f"SPL (A-weighted): {spl_a:.2f} dBSPLA")
                
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Stop and close tasks"""
        if self.input_task:
            self.input_task.stop()
            self.input_task.close()
        if self.output_task:
            self.output_task.stop()
            self.output_task.close()
        plt.close(self.fig)


# Example usage
if __name__ == "__main__":
    plt.ion() 
    analyzer = AudioAnalyzer()
    analyzer.setup_output()
    analyzer.setup_input()
    analyzer.run_measurement()
