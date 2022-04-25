from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
import os


# def test_sound():
#     sound = AudioSegment.from_wav(r'./sound_assets/alarm_one.wav')
#     print('playing a sound...')
#     play(sound)


if __name__ == '__main__':
    # os.chmod('./sound_assets/alarm_one.wav', 777)
    # test_sound()
    print('playing sound')
    playsound('./sound_assets/alarm_one.wav')

