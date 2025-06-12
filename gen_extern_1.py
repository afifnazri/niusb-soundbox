import nidaqmx
import numpy as np
from pydub import AudioSegment
from nidaqmx.constants import AcquisitionType
import os
from pydub import AudioSegment

# Configuration
mp3_path = r"D:\AFIF\WORK\others\mp3\BIRDS OF A FEATHER.mp3"
sample_rate = 48_000  # Standard audio sample rate
output_channel = "Dev1/ao0"  # Analog output channel
voltage_scale = 1.0  # ±5V output range

def play_mp3_through_daq():
    # Load and convert MP3 using pydub
    audio = AudioSegment.from_mp3(mp3_path)
    
    # Convert to mono and set sample rate
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    
    # Get raw samples and normalize
    samples = np.array(audio.get_array_of_samples())
    samples = samples / (2**(8*audio.sample_width-1))  # Normalize to [-1, 1]
    
    # Scale to DAQ voltage range
    samples = samples * voltage_scale
    
    # Configure and play through DAQ
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(output_channel, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=len(samples)
        )
        
        print(f"Playing: {os.path.basename(mp3_path)} (Duration: {len(samples)/sample_rate:.2f}s)")
        task.write(samples, auto_start=False)
        task.start()
        task.wait_until_done()
        # task.wait_until_done(timeout=len(samples) / sample_rate + 10)

    
    print("Playback complete")

if __name__ == "__main__":
    # Set FFmpeg path (required for MP3 support)
    AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"  
    
    play_mp3_through_daq()



