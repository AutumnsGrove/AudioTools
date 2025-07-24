from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf

pipeline = KPipeline(lang_code='b')  # a is american, b is british

text = "becomes the world around, end when a Make sense in actually and enduring."

generator = pipeline(
    text, voice='bf_lily',  # <= change voice here
    speed=1, split_pattern=r'\n+'
)

for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs)  # gs => graphemes/text
    print(ps)  # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i == 0))
    sf.write(f'{i}.wav', audio, 24000)  # save each audio file
