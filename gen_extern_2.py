import nidaqmx
import numpy as np
from pydub import AudioSegment
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import os


class GenExtern2:
    def __init__(self):
        self.mp3_path_ao0 = r"D:\AFIF\WORK\others\mp3\noise\shopping mall.mp3"      
        self.mp3_path_ao1 = r"D:\AFIF\WORK\others\mp3\BIRDS OF A FEATHER.mp3" 
        self.sample_rate = 96_000 
        self.voltage_scales = {'ao0': 0.25, 'ao1': 0.25}  # Separate scales for each channel
        self.device_name = "Dev1"
        self.active_channels = {'ao0': True, 'ao1': True}  # Default: both channels active
        # AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

    def configure(self, mp3_path_ao0=None, mp3_path_ao1=None, sample_rate=None, 
                voltage_scale_ao0=None, voltage_scale_ao1=None, device_name=None, 
                active_channels=None):
        """Update configuration parameters"""
        if mp3_path_ao0: self.mp3_path_ao0 = mp3_path_ao0
        if mp3_path_ao1: self.mp3_path_ao1 = mp3_path_ao1
        if sample_rate: self.sample_rate = sample_rate
        if voltage_scale_ao0 is not None:
            self.voltage_scales['ao0'] = voltage_scale_ao0
        if voltage_scale_ao1 is not None:
            self.voltage_scales['ao1'] = voltage_scale_ao1
        if device_name: self.device_name = device_name
        if active_channels: self.active_channels = active_channels

    # def load_and_normalize_audio(self, mp3_path):
    def load_and_normalize_audio(self, mp3_path, channel):  # Added channel parameter
        """Load and process a single audio file"""
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
            samples = np.array(audio.get_array_of_samples())
            samples = samples / (2 ** (8 * audio.sample_width - 1))
            # return samples * self.voltage_scales
            scale_value = self.voltage_scales[channel]  # Now channel is defined
            return samples * scale_value

        except Exception as e:
            print(f"Error loading {mp3_path}: {str(e)}")
            return None
        
    def play(self):
        # Load audio files only for active channels
        samples_ao0 = self.load_and_normalize_audio(self.mp3_path_ao0,'ao0') if self.active_channels.get('ao0', False) else None
        samples_ao1 = self.load_and_normalize_audio(self.mp3_path_ao1,'ao1') if self.active_channels.get('ao1', False) else None
        if samples_ao0 is None and samples_ao1 is None:
            print("Error: No active channels or failed to load audio files")
            return False

        # Determine maximum length if we have both channels
        max_len = 0
        if samples_ao0 is not None and samples_ao1 is not None:
            max_len = max(len(samples_ao0), len(samples_ao1))
        elif samples_ao0 is not None:
            max_len = len(samples_ao0)
        else:
            max_len = len(samples_ao1)

        # Prepare audio data for active channels
        audio_data = []
        channel_names = []
        
        if self.active_channels.get('ao0', False):
            if samples_ao0 is not None:
                samples_ao0 = np.pad(samples_ao0, (0, max_len - len(samples_ao0)))
                audio_data.append(samples_ao0)
                channel_names.append(f"AO0: {os.path.basename(self.mp3_path_ao0)}")
            else:
                # If channel is active but loading failed, create silent channel
                audio_data.append(np.zeros(max_len))
                channel_names.append("AO0: [silent]")

        if self.active_channels.get('ao1', False):
            if samples_ao1 is not None:
                samples_ao1 = np.pad(samples_ao1, (0, max_len - len(samples_ao1)))
                audio_data.append(samples_ao1)
                channel_names.append(f"AO1: {os.path.basename(self.mp3_path_ao1)}")
            else:
                # If channel is active but loading failed, create silent channel
                audio_data.append(np.zeros(max_len))
                channel_names.append("AO1: [silent]")

        audio_data = np.vstack(audio_data) if len(audio_data) > 1 else np.array(audio_data)

        # DAQmx configuration
        try:
            with nidaqmx.Task() as task:
                # Add output channels based on active channels
                if self.active_channels.get('ao0', False):
                    task.ao_channels.add_ao_voltage_chan(
                        f"{self.device_name}/ao0", min_val=-10.0, max_val=10.0)
                
                if self.active_channels.get('ao1', False):
                    task.ao_channels.add_ao_voltage_chan(
                        f"{self.device_name}/ao1", min_val=-10.0, max_val=10.0)

                # Configure timing
                task.timing.cfg_samp_clk_timing(
                    rate=self.sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=max_len
                )

                # Write and play audio
                writer = AnalogMultiChannelWriter(task.out_stream)
                writer.write_many_sample(audio_data)

                # Print playback info
                for channel in channel_names:
                    print(f"Playing on {channel}")
                print(f"Duration: {max_len / self.sample_rate:.2f}s")

                task.start()
                task.wait_until_done(timeout=max_len / self.sample_rate + 10)
            
            print("Playback completed successfully")
            return True
            
        except Exception as e:
            print(f"DAQmx error: {str(e)}")
            return False


if __name__ == "__main__":
    # When run directly, install dependencies if needed
    try:
        from pydub import AudioSegment
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "pydub", "nidaqmx", "numpy"])
    
    # Example usage
    player = GenExtern2()
    
    # # Play only AO0
    # player.configure(active_channels={'ao0': True, 'ao1': False})
    # player.play()
    
    # # Play only AO1
    player.configure(
        active_channels={
            'ao0': False,
            'ao1': True,
            }
            )
    player.play()
    
    # # Play both channels (default)
    # player.configure(active_channels={'ao0': True, 'ao1': True})
    # player.play()