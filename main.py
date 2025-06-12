import threading
import time

from nidaqmx.constants import AcquisitionType

from gen_extern_2 import GenExtern2
from spl_fft_A import SPL
from gen_sine import GenSine


def main():
    # Initialize signal generator with custom parameters (optional)
    sig_gen = GenSine(device_name="Dev1")
    sig_gen.frequencies = [1000, 5000]  # 1 kHz and 5 kHz
    # sig_gen.amplitude = 0.25  # 1V amplitude
    sig_gen.amplitude = 1  # 1V amplitude
    sig_gen.sample_rate = 96_000  # 50 kHz sample rate

    try:
        # Start signal generation (0.5 second buffer)
        sig_gen.start_generation(duration=0.5)
        
        # Main loop - keep running until KeyboardInterrupt
        print("Signal generation running...")
        while True:
            time.sleep(0.1)  # Reduce CPU usage
            
            # You could add runtime monitoring/control here
            # Example:
            # if some_condition:
            #     sig_gen.stop_generation()
            #     break
            
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt...")
    finally:
        # Ensure resources are always cleaned up
        sig_gen.stop_generation()
        print("Program exited cleanly")




#     """thread play signal & measure SPL"""
#     # slm = SPL()
#     # player = GenExtern2()
#     # player.configure(
#     #     voltage_scale_ao0=1.0,  
#     #     voltage_scale_ao1=0.25,
#     #     active_channels={
#     #         'ao1': True,    #song
#     #     }
#     # )

#     # try:
#     #     play_thread = threading.Thread(target=player.play)
#     #     measure_thread = threading.Thread(target=slm.run_measurement)

#     #     play_thread.start()
#     #     measure_thread.start()

#     #     play_thread.join()
#     #     measure_thread.join()

#     # except Exception as e:
#     #     print(f"Error: {e}")
#     # finally:
#     #     slm.close()

#     """Measure SPL"""
#     # slm = SPL()
#     # try:
#     #     slm.run_measurement()
#     # except Exception as e:
#     #     print(f"Error: {e}")
#     # finally:
#     #     slm.close()


#     """Genertae signal"""
#     # player = GenExtern2()
#     # player.configure(
#     #     # mp3_path_ao0=r"path\to\your\first_file.mp3",
#     #     # mp3_path_ao1=r"path\to\your\second_file.mp3",
#     #     # sample_rate=96_000,
#     #     voltage_scale_ao0=1.0,  
#     #     voltage_scale_ao1=0.25,
#     #     active_channels={
#     #         # 'ao0': True,    #noise
#     #         'ao1': True,    #song
#     #         }
#     #     )
    
#     # success = player.play()
#     # if not success:
#     #     print("Playback failed")

# if __name__ == "__main__":
#     main()


