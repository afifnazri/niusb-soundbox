import numpy as np

class MockTask:
    def __init__(self):
        self.read_calls = 0

    def ai_channels(self):
        return self

    def add_ai_voltage_chan(self, *args, **kwargs):
        pass

    def timing(self):
        return self

    def cfg_samp_clk_timing(self, *args, **kwargs):
        pass

    def read(self, number_of_samples_per_channel):
        # Simulate a sine wave + noise as input data
        t = np.linspace(0, 1, number_of_samples_per_channel, endpoint=False)
        # signal = 0.05 * np.sin(2 * np.pi * 1000 * t) + 0.005 * np.random.randn(len(t))
        signal = 10 * np.random.randn(len(t))

        return signal.tolist()  # Simulate the DAQ return format

    def close(self):
        pass
