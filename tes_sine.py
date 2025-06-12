import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt
import time

class SineWaveGenerator:
    def __init__(self, device_name='Dev1', sample_rate=100_000, amplitude=1.0, frequency=1.0):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.frequency = frequency
        self.task = None
        self.running = False
        
    def generate_sine_wave(self, duration=None, output_channel='ao0'):
        """Generate a continuous sine wave. If duration is None, runs until stopped."""
        
        buffer_duration = 0.1  # seconds of data in each buffer
        samples_per_buffer = int(self.sample_rate * buffer_duration)
        num_samples = samples_per_buffer if duration is None else int(self.sample_rate * duration)
        
        # Generate time array and sine wave data for one buffer
        t = np.linspace(0, buffer_duration, samples_per_buffer, endpoint=False)
        sine_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
        try:
            self.task = nidaqmx.Task()
            self.task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{output_channel}",
                min_val=-10.0,
                max_val=10.0
            )
            
            # Configure for continuous generation
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=samples_per_buffer
            )
            
            # Write the initial data
            self.task.write(sine_wave)
            
            print(f"Generating {self.frequency} Hz continuous sine wave...")
            self.running = True
            self.task.start()
            
            if duration is not None:
                # For finite operation
                time.sleep(duration)
                self.stop()
            else:
                # For continuous operation until stopped
                while self.running:
                    # Continually write the same data in a loop
                    self.task.write(sine_wave)
                    time.sleep(buffer_duration * 0.9)  # Slightly less than buffer duration
                    
        except Exception as e:
            print(f"Error: {e}")
            self.stop()
            
    def stop(self):
        """Stop the continuous generation"""
        if self.task is not None:
            self.running = False
            self.task.stop()
            self.task.close()
            self.task = None
            print("Generation stopped.")


# Example usage
if __name__ == "__main__":
    # Create sine wave generator instance
    swg = SineWaveGenerator(device_name='Dev1',
                           sample_rate=200_000,
                           amplitude=1.0,
                           frequency=20_000)
    
    try:
        # Start continuous generation (will run until stopped)
        swg.generate_sine_wave()
        
        # Let it run for 5 seconds (or press Ctrl+C to stop)
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("Stopping by user request...")
    finally:
        swg.stop()
        print("Program completed.")