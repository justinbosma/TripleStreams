import pretty_midi

def remove_small_intervals(midi):
    min_interval = 1
    intervals = []
    epsilon = 1e-3
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end - note.start <= epsilon:
                print("error")
                instrument.notes.remove(note)
            else:
            # print("ok")
                #intervals.append(note.end - note.start)
                #min_interval = min(min_interval, note.end - note.start)
                pass