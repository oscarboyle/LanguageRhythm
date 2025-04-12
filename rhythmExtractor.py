import music21 as m
from music21.analysis.patel import nPVI
import numpy as np
import fasttext
import json
from tqdm import tqdm
import os
import pandas as pd


def get_language(score,language_model):
    """ 
    Extracts lyrics from music21 score and predicts language
    using fasttext model.
    Args:
        score (music21.stream.Score): The music21 score object.
        language_model (fasttext.FastText._FastText): The pretrained fasttext model.
    Returns:
        str: The predicted language of the lyrics.
    """
    # Extract lyrics
    lyrics = []

    for element in score.recurse():
        if isinstance(element, m.note.Note) and element.lyrics:
            for lyric in element.lyrics:
                lyrics.append(lyric.text)

    if not lyrics:
        return "No Lyrics"
    
    # Join all lyric syllables/words
    full_lyrics = ' '.join(lyrics)

    # Download the pretrained model from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    model = language_model
    predictions = model.predict(full_lyrics, k=1)  # Get top 3 predictions

    #Get language name from iso json
    iso_file = 'misc/languageISO.json'

    with open(iso_file, 'r', encoding='utf-8') as f:
        iso_data = json.load(f)

    # Convert the prediction to language name

    label = predictions[0][0].replace('__label__', '')
    language = iso_data.get(label, 'Unknown Language')

    return language



def get_nPVI(score):
    """
    Calculate the normalized pairwise variability index for the melody of a score.
    0~20 is considered low variability
    20~40 is considered moderate
    >60 means more rhythmic contrast or irregularity

    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        float: The nPVI value for the melody.
    """
    # Extract melody notes
    melody_notes = [el for el in score.recurse() if isinstance(el, m.note.Note)]

    if not melody_notes:
        return None

    # Put notes into a new stream
    melody_stream = m.stream.Stream(melody_notes)

    # Calculate melodic interval variability (nPVI)
    nPVI_value = nPVI(melody_stream)

    return nPVI_value

def get_inter_onset_interval(score):
    """
    Calculates the interonset intervals of the melody of a score

    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        float: The nPVI value for the melody.
    """
    melody = score.parts[0].flatten().recurse().notes

    # Check if the score has a melody part
    if not melody:
        return [], None, None
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

def get_rhythmic_density(score):
    """
    Calculates rhuthmic density of melody in score
    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        float: density of score
        float: number of notes in score
        float: totlal duration of score
    """
    # Get the main melody line
    melody = score.parts[0].flatten()
    notes = [n for n in melody if isinstance(n, m.note.Note)]

    if len(notes) < 2:
        return None, None, None  # not enough data

    # Get total duration of the part (in quarter lengths)
    total_duration = melody.highestTime

    # Get number of notes
    num_notes = len(notes)

    # Compute density
    rhythmic_density = num_notes / total_duration if total_duration > 0 else 0

    return rhythmic_density, num_notes, total_duration

def extract_onset_times(score):
    """
    Extract onset times (in beats) from a MusicXML file.
    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        list: List of onset times in beats
    """
    part = score.parts[0]  # Get the first part (melody)
    
    onset_times = []
    current_time = 0  # Track onset times in beats
    
    for note in part.flat.notes:
        onset_times.append(current_time)
        current_time += note.quarterLength  # Advance time by note duration
    
    return onset_times

def compute_autocorrelation_periodicity(onsets):
    """
    Compute periodicity using autocorrelation of inter-onset intervals (IOIs)
    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        integer: dominant period of the score in beats
        
    """
    iois = np.diff(onsets)  # Compute inter-onset intervals (IOIs)
    
    # Compute autocorrelation
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags

    # Find first peak (excluding zero lag)
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
    if len(peaks) == 0:
        return None

    dominant_period = peaks[0]  # First significant peak
    return dominant_period * np.mean(iois) # Convert to beats


def lempel_ziv_complexity(seq):
    """
    Calculate the lempel-ziv complexity of sequence
    Args:
        seq (list): Sequence of elements to analyze
    Returns:
        int: Lempel-Ziv complexity of the sequence
    """
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


def get_rhythm_sequence(score):
    """
    Extracts rhythm sequence (note durations) from the melody of the score.
    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        list: list of note durations in quarter lenght     
    """
    melody = score.parts[0].flatten()
    notes = [n for n in melody if isinstance(n, m.note.Note)]

    if not notes:
        return []

    # You could use n.quarterLength or round if desired
    rhythm_sequence = [round(n.quarterLength, 3) for n in notes]
    return rhythm_sequence


def get_lz_complexity_from_score(score):
    """
    Calculates Lempel-Ziv complexity of rhythmic sequence from a score, using
    get_rhythm_sequence and lempel_ziv_complexity functions.
    Args:
        score (music21.stream.Score): The music21 score object.
    Returns:
        float: Lempel-Ziv complexity of the rhythmic sequence
        float: normalized Lempel-Ziv complexity of the rhythmic sequence
    """
    rhythm_seq = get_rhythm_sequence(score)
    if not rhythm_seq:
        return "No Notes"
    
    lz = lempel_ziv_complexity(rhythm_seq)
    norm_lz = lz / len(rhythm_seq) if len(rhythm_seq) > 0 else 0
    return lz, norm_lz

def get_rhythm_features(file,model):
    """
    Computes all rhythmic features from a file
    Accepts .mxl and .krn files
    Args:
        file (str): Path to .mxl or .krn files
        model (fasttext.FastText._FastText): The pretrained fasttext model.
    Returns:
        dict: Dictionary with rhythmic features
    """
    # Load the .mxl file
    try:
        score = m.converter.parse(file)  # path to your .mxl
        melody = score.parts[0].flatten()
        notes = [n for n in melody if isinstance(n, m.note.Note)]
        # Check if the score has a melody part
        if not melody:
            return None
        #Check lenght of melody
        if len(notes) < 2:
            return None
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return None

    # Get language of lyrics
    language = get_language(score,model)
    # Get rhythmic density
    rhythmic_density, num_notes, total_duration = get_rhythmic_density(score)

    # Get inter-onset intervals
    io_intervals, avg_ioi, stdev_ioi = get_inter_onset_interval(score)

    # Get melodic variability
    nPVI_value = get_nPVI(score)

    # Extract onset times
    onsets = extract_onset_times(score)

    # Compute periodicity using autocorrelation
    periodicity = compute_autocorrelation_periodicity(onsets)

    # Compute Lempel-Ziv complexity
    lz, norm_lz = get_lz_complexity_from_score(score)

    return {
        "language": language,
        "rhythmic_density": rhythmic_density,
        "num_notes": num_notes,
        "total_duration": total_duration,
        "avg_ioi": avg_ioi,
        "stdev_ioi": stdev_ioi,
        "nPVI": nPVI_value,
        "periodicity": periodicity,
        "lz_complexity": lz,
        "norm_lz_complexity": norm_lz
    }
    

if __name__ == "__main__":


    data_csv = 'data/OpenEWLD_data.csv'
    database_folder = '/home/usuari/Desktop/SMC-Master/AMPLab/LanguageRhythm_DATABASES/OpenEWLD' #sometimes the database is in subfolder
    feature_folder = 'features/'

    ##### Download pretrained model from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    model = fasttext.load_model('misc/lid.176.bin')
    dataset_df = pd.read_csv(data_csv)

    tqdm.pandas()

    # Apply directly with a lambda inside progress_apply
    features_df = dataset_df.progress_apply(
        lambda row: pd.Series(get_rhythm_features(
            os.path.join(database_folder, row['path'].lstrip('/'))  # clean path
            ,model
        )) if os.path.exists(os.path.join(database_folder, row['path'].lstrip('/')))
        else pd.Series({
            "language": None,
            "rhythmic_density": None,
            "num_notes": None,
            "total_duration": None,
            "avg_ioi": None,
            "stdev_ioi": None,
            "nPVI": None,
            "periodicity": None,
            "lz_complexity": None,
            "norm_lz_complexity": None
        }),
        axis=1
    )

    # Combine original data with extracted features
    full_df = pd.concat([dataset_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Save the output
    os.makedirs(feature_folder, exist_ok=True)
    output_path = os.path.join(feature_folder, 'OpenEWLD_rhythmic_features.csv')
    full_df.to_csv(output_path, index=False)
