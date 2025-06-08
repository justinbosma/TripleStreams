import matplotlib.pyplot as plt
import pretty_midi
import os

def plot_instrument_histogram(directory):
        instrument_types_for_hist = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                pm_hist = pretty_midi.PrettyMIDI(os.path.join(root, file))
                for instrument in pm_hist.instruments:
                    instrument_types_for_hist.append(instrument.name)
        plt.hist(instrument_types_for_hist, edgecolor='black')
        plt.xticks(rotation=90)
        plt.show()

def main(args):
    if len(args) != 2:
        print("Usage: python plot_instrument_histogram.py <directory>")
        return
    directory = args[1]
    plot_instrument_histogram(directory)

if __name__ == "__main__":
    import sys
    main(sys.argv)