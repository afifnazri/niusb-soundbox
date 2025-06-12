import nidaqmx
import numpy as np
import time
from nidaqmx.constants import AcquisitionType, RegenerationMode
from nidaqmx.stream_writers import AnalogMultiChannelWriter


class GenSine:
    def __init__(self, sample_rate=None, frequency_ao0=None,frequency_ao1=None, amplitude=None,duration=None):
        self.sample_rate = sample_rate if sample_rate is not None else 200_000  # Hz
        self.frequency_ao0 = frequency_ao0 if frequency_ao0 is not None else 1_000  # Hz
        self.frequency_ao1 = frequency_ao1 if frequency_ao1 is not None else 1_000  # Hz
        self.amplitude = amplitude if amplitude is not None else 0.5  # Volts
        self.duration = duration if duration is not None else 0.5  # seconds

        # Generate time and waveforms
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        waveform_ao0 = self.amplitude * np.sin(2 * np.pi * self.frequency_ao0 * t)
        waveform_ao1 = self.amplitude * np.sin(2 * np.pi * self.frequency_ao1 * t)

        # Stack into 2D array (2 channels x N samples)
        self.data = np.vstack((waveform_ao0, waveform_ao1))

    def start_output(self):
        with nidaqmx.Task() as task:
            # Add AO channels
            task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
            task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)

            task.timing.cfg_samp_clk_timing(rate=self.sample_rate,sample_mode=AcquisitionType.CONTINUOUS)
            task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
            writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)
            writer.write_many_sample(self.data)  # Write once
            task.start()

            try:
                while True:
                    time.sleep(0.5)  # No need to write again

            except KeyboardInterrupt:
                print("Stopped by user.")

            finally:
                task.stop()


if __name__ == "__main__":
    generator = GenSine(
        sample_rate=200_000,
        frequency_ao0=1000,
        frequency_ao1=20_000,
        amplitude=1.0,
    )
    generator.start_output()



## non continous ##
# import nidaqmx
# import numpy as np
# from nidaqmx.constants import AcquisitionType

# # Parameters
# sample_rate = 200_000  # Hz
# duration = 10  # seconds
# frequency_ao0 = 20_000   # Hz (1 kHz)
# frequency_ao1 = 10_000  # Hz (10 kHz)
# amplitude = 1  # Volts

# # Generate waveforms
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# waveform_ao0 = amplitude * np.sin(2 * np.pi * frequency_ao0 * t)
# waveform_ao1 = amplitude * np.sin(2 * np.pi * frequency_ao1 * t)
# data = np.vstack((waveform_ao0, waveform_ao1))  # shape: (2, N)

# with nidaqmx.Task() as task:
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)

#     task.timing.cfg_samp_clk_timing(rate=sample_rate,
#                                      sample_mode=AcquisitionType.FINITE,
#                                      samps_per_chan=data.shape[1])

#     from nidaqmx.stream_writers import AnalogMultiChannelWriter
#     writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)
#     writer.write_many_sample(data)

#     task.start()
#     task.wait_until_done(timeout=duration + 1)



# continous ##

# import nidaqmx
# import numpy as np
# import time
# from nidaqmx.constants import AcquisitionType
# from nidaqmx.stream_writers import AnalogMultiChannelWriter

# # Parameters
# sample_rate = 100_000  # Hz
# duration = 0.5  # seconds
# frequency_ao0 = 1_000  # Hz (1 kHz for AO0)
# frequency_ao1 = 1_000  # Hz (10 kHz for AO1)
# amplitude = 1  # Volts

# # Generate time and waveforms
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# waveform_ao0 = amplitude * np.sin(2 * np.pi * frequency_ao0 * t)
# waveform_ao1 = amplitude * np.sin(2 * np.pi * frequency_ao1 * t)

# # Stack into 2D array (2 channels x N samples)
# data = np.vstack((waveform_ao0, waveform_ao1))

# with nidaqmx.Task() as task:

#     task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao1", min_val=-10.0, max_val=10.0)
#     task.timing.cfg_samp_clk_timing(rate=sample_rate,sample_mode=AcquisitionType.CONTINUOUS)
#     writer = AnalogMultiChannelWriter(task.out_stream, auto_start=True)
#     try:
#         while True:
#             writer.write_many_sample(data)  # data shape: (num_channels, num_samples) 
#             time.sleep(duration / 2)  # Slightly faster than playback time

#     except KeyboardInterrupt:
#         print("Stopped by user.")


## class ##

