import time
from playsound import playsound

VIOLATION_CLASSES = [ # to be updated if meron nang other classes\
    'no face_shield',
    "no face_mask"
]



def sound_signal():
    playsound('./sound_assets/alarm_one.wav')

def check_if_violates_any(predictions):
    for prediction in predictions:
        if prediction[0] in VIOLATION_CLASSES:
            return True
    return False

def count_nomask_violations(predictions):
    num_instances = 0
    for prediction in predictions:
        if prediction[0] == 'no face_mask':
            num_instances += 1
    return str(num_instances)

def count_noshields_violations(predictions):
    num_instances = 0
    for prediction in predictions:
        if prediction[0] == 'no face_shield':
            num_instances += 1
    return str(num_instances)

class TimeForSoundChecker:

    def __init__(self):
        self.time_last_called = time.time() * 1000

    def has_been_a_second(self):
        current = time.time() * 1000
        if (current - self.time_last_called) > 1000:
            self.time_last_called = current
            return True
        return False

