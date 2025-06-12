import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import bilinear, lfilter


class SPL:
    def __init__(self, sample_rate=96000, samples_per_read=32*1024):
        # DAQ Settings
        self.SAMPLE_RATE = sample_rate
        self.SAMPLES_PER_READ = samples_per_read

        # Microphone calibration parameters
        self.ref_Pa = 20e-6
        self.mic_sensitivity = 12e-3
        self.preamp_gain = 21
        # self.mic_sensitivity = 14e-3
        # preamp_gain = 14.7
        # mic_sensitivity = 17e-3

        # Initialize A-weighting filter coefficients
        self.b, self.a = self._a_weighting(self.SAMPLE_RATE)

        # Setup DAQ task
        self.task = None
        self._setup_daq()
        
        # # Initialize plot
        self.fig, self.ax = None, None
        self.line = None
        self._setup_plot()


    def _a_weighting(self, fs):
        """Calculate A-weighting filter coefficients."""
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

    def _setup_daq(self):
        """Configure the DAQ task."""
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0",
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-1,
            max_val=1
        )

        self.task.timing.cfg_samp_clk_timing(
            rate=self.SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.SAMPLES_PER_READ
        )


    # def _setup_plot(self):
    #         """Initialize the frequency spectrum plot."""
    #         plt.ion()
    #         self.fig, self.ax = plt.subplots()
    #         x = np.fft.rfftfreq(self.SAMPLES_PER_READ, 1/self.SAMPLE_RATE)
    #         self.line, = self.ax.plot(x, np.zeros_like(x))
            
    #         self.ax.set_xscale('log')
    #         self.ax.set_xlim(20, 20000)
    #         self.ax.set_ylim(-200, 0)
    #         self.ax.set_xlabel('Frequency (Hz)')
    #         self.ax.set_ylabel('Magnitude (dB)')
    #         self.ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    #         self.ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    #         self.ax.grid(True, which='both', ls='--')
    #         plt.show()
    
    def _calculate_spl(self, data):
        """Calculate sound pressure level from raw data."""
        rms_voltage = (np.sqrt(np.mean(data**2)))/2
        voltage_mic = rms_voltage/(10**(self.preamp_gain/20))
        mic_Pa = voltage_mic/self.mic_sensitivity
        spl = 20 * np.log10(mic_Pa/(self.ref_Pa))
        return spl, rms_voltage, voltage_mic
    
    def _calculate_spl_a(self, data):
        """Calculate A-weighted sound pressure level."""
        weighted_data = lfilter(self.b, self.a, data)
        windowed = weighted_data * np.hanning(len(weighted_data))
        
        rms_voltage_a = np.sqrt(np.mean(windowed**2)) / 2
        mic_voltage_a = rms_voltage_a / (10**(self.preamp_gain / 20))
        mic_pa_a = mic_voltage_a / self.mic_sensitivity
        spl_a = 20 * np.log10(mic_pa_a / self.ref_Pa)
        
        return spl_a, windowed
    
    # def _update_plot(self, windowed_data):
    #     """Update the frequency spectrum plot."""
    #     fft_vals = np.fft.rfft(windowed_data)
    #     mag = np.abs(fft_vals) / self.SAMPLES_PER_READ
    #     mag_db = 20 * np.log10(mag + 1e-12)
        
        # x = np.fft.rfftfreq(self.SAMPLES_PER_READ, 1/self.SAMPLE_RATE)
        # freq_mask = (x >= 20) & (x <= 20000)
        
        # self.line.set_xdata(x[freq_mask])
        # self.line.set_ydata(mag_db[freq_mask])
        # self.ax.relim()
        # self.ax.autoscale_view(True, True, True)
        # plt.pause(0.01)
    
    def run_measurement(self):
        print("Reading RMS voltage continuously. Press Ctrl+C to stop.")
        try:
            while True:
                # Read data from DAQ
                data = self.task.read(number_of_samples_per_channel=self.SAMPLES_PER_READ)
                np_data = np.array(data)
                
                # Calculate SPL
                spl, rms_voltage, voltage_mic = self._calculate_spl(np_data)
                
                # Calculate A-weighted SPL and get windowed data for FFT
                spl_a, windowed_data = self._calculate_spl_a(np_data)
                
                # Update plot
                self._update_plot(windowed_data)
                
                # Print results
                print(f"Mic voltage:{voltage_mic:.3f} V | RMS Voltage: {rms_voltage:.6f} V | SPL: {spl:.2f}dBSPL | SPL A: {spl_a:.2f}dBSPLA")
                
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            self.close()
    
    def close(self):
        """Clean up resources."""
        if self.task is not None:
            self.task.close()
        plt.close(self.fig)


if __name__ == "__main__":
    # Example usage
    spl = SPL()
    spl.run_measurement()




# import nidaqmx
# from nidaqmx.constants import AcquisitionType, TerminalConfiguration
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from scipy.signal import bilinear, lfilter


# # DAQ Settings
# SAMPLE_RATE = 96_000
# SAMPLES_PER_READ = 64 * 1024  

# ref_Pa = 20e-6
# mic_sensitivity = 14e-3
# fs_vpeak = 1.712
# preamp_gain = 21
# # preamp_gain = 14.7
# # mic_sensitivity = 17e-3

# def a_weighting(fs):
#     f1 = 20.598997
#     f2 = 107.65265
#     f3 = 737.86223
#     f4 = 12194.217
#     A1000 = 1.9997

#     NUMs = [(2 * np.pi * f4)**2 * (10**(A1000 / 20)),
#             0, 0, 0, 0]
#     DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
#                        [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
#     DENs = np.convolve(np.convolve(DENs,[1, 2 * np.pi * f3]),
#                        [1, 2 * np.pi * f2])

#     b, a = bilinear(NUMs, DENs, fs)
#     return b, a
# b, a = a_weighting(SAMPLE_RATE)


# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan(
#         "Dev1/ai0",
#         terminal_config=TerminalConfiguration.DIFF,
#         min_val=-1,
#         max_val=1
#     )

#     task.timing.cfg_samp_clk_timing(
#         rate=SAMPLE_RATE,
#         sample_mode=AcquisitionType.CONTINUOUS,
#         # samps_per_chan=SAMPLES_PER_READ * 2  # Buffer size
#         samps_per_chan=SAMPLES_PER_READ * 1  # Bigger buffer

#     )

#     print("Reading RMS voltage continuously. Press Ctrl+C to stop.")

#     plt.ion()
#     fig, ax = plt.subplots()
#     x = np.fft.rfftfreq(SAMPLES_PER_READ, 1/SAMPLE_RATE)
#     line, = ax.plot(x, np.zeros_like(x))  # Use plot, not semilogx

#     ax.set_xscale('log')  # Set log scale manually
#     ax.set_xlim(20, 20000)
#     ax.set_ylim(-200, 0)
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Magnitude (dB)')
#     ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show full numbers, no scientific notation
#     ax.grid(True, which='both', ls='--')

#     plt.show()


#     try:
#         while True:
#             data = task.read(number_of_samples_per_channel=SAMPLES_PER_READ)
#             np_data = np.array(data)

#             rms_voltage = (np.sqrt(np.mean(np_data**2)))/2
#             voltage_mic = rms_voltage/(10**(preamp_gain/20))
#             mic_Pa = voltage_mic/mic_sensitivity
#             spl = 20 * np.log10 (mic_Pa/(ref_Pa))


#             weighted_data = lfilter(b, a, np_data)
#             windowed = weighted_data * np.hanning(len(weighted_data))

#             rms_voltage_a = np.sqrt(np.mean(windowed**2)) / 2
#             mic_voltage_a = rms_voltage_a / (10**(preamp_gain / 20))
#             mic_pa_a = mic_voltage_a / mic_sensitivity
#             spl_a = 20 * np.log10(mic_pa_a / ref_Pa)

#             # FFT magnitude
#             fft_vals = np.fft.rfft(windowed)
#             mag = np.abs(fft_vals) / SAMPLES_PER_READ  # Normalize
#             mag_db = 20 * np.log10(mag + 1e-12)  # Convert to dB, avoid log(0)

#             # Limit FFT to 20Hz-20kHz
#             freq_mask = (x >= 20) & (x <= 20000)
#             line.set_xdata(x[freq_mask])
#             line.set_ydata(mag_db[freq_mask])
#             ax.relim()
#             ax.autoscale_view(True, True, True)
#             plt.pause(0.01)


#             print(f"Mic voltage:{voltage_mic:.3f} V | RMS Voltage: {rms_voltage:.6f} V | SPL: {spl:.2f}dBSPL |  SPL A: {spl_a:.2f}dBSPLA")


#     except KeyboardInterrupt:
#         print("Stopped by user.")