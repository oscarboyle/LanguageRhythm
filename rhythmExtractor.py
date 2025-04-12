import music21 as m
from music21.analysis.patel import melodicIntervalVariability as npVI
import numpy as np


def get_melody_variability(file):
    # Load the .mxl file
    score = m.converter.parse(file)

    # Extract melody notes
    melody_notes = [el for el in score.recurse() if isinstance(el, m.note.Note)]

    if not melody_notes:
        return "No Melody"

    # Put notes into a new stream
    melody_stream = m.stream.Stream(melody_notes)

    # Calculate melodic interval variability (nPVI)
    variability = npVI(melody_stream)

    return variability

def get_inter_onset_interval(file):
    score = m.converter.parse(file)

    melody = score.parts[0].flatten().recurse().notes

    # Check if the score has a melody part
    if not melody:
        return "No Melody"
    # Sort notes by onset (offset) to prevent negative IOIs
    notes = sorted(melody, key=lambda n: n.offset)

    if not notes:
        return "No Notes"


    io_intervals = [
        notes[i].offset - notes[i - 1].offset for i in range(1, len(notes))
    ]

    avg_ioi = sum(io_intervals) / len(io_intervals)
    stdev_ioi = (sum((x - avg_ioi) ** 2 for x in io_intervals) / len(io_intervals)) ** 0.5

    return io_intervals, avg_ioi, stdev_ioi

def get_rhythmic_density(file):
    score = m.converter.parse(file)

    # Get the main melody line
    melody = score.parts[0].flatten()
    notes = [n for n in melody if isinstance(n, m.note.Note)]

    if not notes:
        return "No Notes"

    # Get total duration of the part (in quarter lengths)
    total_duration = melody.highestTime

    # Get number of notes
    num_notes = len(notes)

    # Compute density
    rhythmic_density = num_notes / total_duration if total_duration > 0 else 0

    return rhythmic_density, num_notes, total_duration

def extract_onset_times(musicxml_file):
    """Extract onset times (in beats) from a MusicXML file."""
    score = m.converter.parse(musicxml_file)
    part = score.parts[0]  # Get the first part (melody)
    
    onset_times = []
    current_time = 0  # Track onset times in beats
    
    for note in part.flat.notes:
        onset_times.append(current_time)
        current_time += note.quarterLength  # Advance time by note duration
    
    return onset_times

def compute_autocorrelation_periodicity(onsets):
    """Compute periodicity using autocorrelation of inter-onset intervals (IOIs)."""
    iois = np.diff(onsets)  # Compute inter-onset intervals (IOIs)
    
    # Compute autocorrelation
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags

    # Find first peak (excluding zero lag)
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
    if len(peaks) == 0:
        print("No clear periodicity found.")
        return None

    dominant_period = peaks[0]  # First significant peak
    return dominant_period


def lempel_ziv_complexity(seq):
    n = len(seq)
    if n == 0:
        return 0

    i = 0
    c = 1
    parsed_subs = set()
    w = [seq[0]]

    while i + 1 < n:
        w.append(seq[i + 1])
        if tuple(w) not in parsed_subs:
            parsed_subs.add(tuple(w))
            c += 1
            w = []
            if i + 1 < n:
                w = [seq[i + 1]]
        i += 1
    return c


def get_pitch_sequence(file):
    score = m.converter.parse(file)
    melody = score.parts[0].flatten()
    notes = [n for n in melody if isinstance(n, m.note.Note)]

    if not notes:
        return []

    # Create a sequence of MIDI pitches (or string pitch names if preferred)
    pitch_sequence = [n.pitch.midi for n in notes]
    print(pitch_sequence)
    return pitch_sequence

def get_lz_complexity_from_mxl(file):
    pitch_seq = get_pitch_sequence(file)
    if not pitch_seq:
        return "No Notes"
    
    lz = lempel_ziv_complexity(pitch_seq)
    norm_lz = lz / len(pitch_seq) if len(pitch_seq) > 0 else 0
    return lz,norm_lz

def get_rhythm_features(file):
    # Get rhythmic density
    rhythmic_density, num_notes, total_duration = get_rhythmic_density(file)

    # Get inter-onset intervals
    io_intervals, avg_ioi, stdev_ioi = get_inter_onset_interval(file)

    # Get melodic variability
    melodic_variability = get_melody_variability(file)

    # Extract onset times
    onsets = extract_onset_times(file)

    # Compute periodicity using autocorrelation
    periodicity = compute_autocorrelation_periodicity(onsets)

    # Compute Lempel-Ziv complexity
    lz, norm_lz = get_lz_complexity_from_mxl(file)

    return {
        "rhythmic_density": rhythmic_density,
        "num_notes": num_notes,
        "total_duration": total_duration,
        "avg_ioi": avg_ioi,
        "stdev_ioi": stdev_ioi,
        "melodic_variability": melodic_variability,
        "periodicity": periodicity,
        "lz_complexity": lz,
        "norm_lz_complexity": norm_lz
    }