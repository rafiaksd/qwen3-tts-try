# pip install qwen_tts flash-attnfrom qwen_tts import Qwen3TTSModel
import time 
import torch

start_time = time.time()

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # or 0.6B-Base
   device_map="auto"
)
print("Model loaded...")

ref_audio = "ref.wav"
ref_text = "your text goes here"

gen_text = "text you want TTS to voice"
print("Generating voice")
wavs, sr = model.generate_voice_clone(
   ref_audio=ref_audio,
   ref_text=ref_text,
   text = gen_text
)

import soundfile as sf
sf.write("cloned_speech.wav", wavs[0], sr)

torch.cuda.empty_cache()
print("🗑️🗑️  Reset torch cache...")

print(f"⏰⏰ Time taken: {time.time()-start_time:.2f}s")
