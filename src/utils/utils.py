"""
Module to track the time or progress in our model using a progress bar and
formatting time, som utilities here"
"""
# utils.py

# Standard libraries
from typing import Any, Dict
from typing import Callable, Optional, Tuple
import sys
import os
import time
import random  # Agregar la importación de random

# Third-party libraries
import numpy as np
import pretty_midi
import torch
from torch.autograd import Variable


__all__ = [
    "load_midi_data",
    "load_all_midi_files",
    "progress_bar",
    "logger",
    "format_time",
    "rand_slice",
    "seq_to_tensor",
    "train_slice",
    "train_batch",
    "song_to_seq_target",
    "save_checkpoint",
    "load_checkpoint"
]


def load_midi_data(file_path: str) -> Optional[np.ndarray]:
    """
    Loads a MIDI file and extracts musical features: pitch, step, duration,
    and optional velocity.

    Args:
        file_path (str): The path to the MIDI file.

    Returns:
        Optional[np.ndarray]: A numpy array containing the features of
        each note in the
        format [pitch, step, duration, velocity]. Returns None if the
        file cannot be loaded.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        instrument = midi_data.instruments[0]
        notes = instrument.notes
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Initialize feature lists
    features = []
    prev_start = 0.0

    for note in notes:
        pitch = note.pitch
        start = note.start
        end = note.end
        step = start - prev_start
        duration = end - start
        velocity = note.velocity

        features.append([pitch, step, duration, velocity])
        prev_start = start

    return np.array(features)


def load_all_midi_files(midi_dir_path: str) -> Optional[np.ndarray]:
    """
    Loads and processes all MIDI files in a given directory.

    Args:
        midi_dir_path (str): The directory path where MIDI files are stored.

    Returns:
        Optional[np.ndarray]: Combined musical features from all MIDI files.
    """
    if not os.path.exists(midi_dir_path):
        print(f"Directory does not exist: {midi_dir_path}")
        return None

    midi_files = [f for f in os.listdir(midi_dir_path) if f.endswith('.mid')]
    all_music_data = []

    for midi_file in midi_files:
        midi_file_path = os.path.join(midi_dir_path, midi_file)
        print(f"Loading {midi_file_path}...")

        music_data = load_midi_data(midi_file_path)
        if music_data is not None:
            print(f"Loaded {music_data.shape[0]} notes from {midi_file_path}")
            all_music_data.append(music_data)
        else:
            print(f"Failed to load {midi_file_path}")

    if all_music_data:
        combined_music_data = np.concatenate(all_music_data, axis=0)
        print(f"Total notes loaded from all files: {
              combined_music_data.shape[0]}")
        return combined_music_data
    else:
        print("No music data loaded.")
        return None


def progress_bar(
        current: int,
        start_time: float,
        msg: Optional[str] = None) -> None:
    """
    Displays a progress bar with the time taken per step and the total
    elapsed time.

    Args:
        current (int): The current step in the progress.
        total (int): The total number of steps.
        start_time (float): The time at which the process started.
        msg (Optional[str], optional): An additional message to display.
        Defaults to None.
    """
    cur_time = time.time()
    elapsed_time = cur_time - start_time
    step_time = elapsed_time / (current + 1)  # Avoid division by 0

    progress_message = f"Step: {format_time(step_time)} | Tot: {
        format_time(elapsed_time)}"
    if msg:
        progress_message += f" | {msg}"

    sys.stdout.write(progress_message + '\r')
    sys.stdout.flush()


def logger(verbose: bool = True) -> Callable[[str], None]:
    """
    Creates a logging function that prints messages to the console if verbose
    is True.

    Args:
        verbose (bool, optional): If True, the logger will print messages.
        Defaults to True.

    Returns:
        Callable[[str], None]: A function that takes a message
        (or messages) and prints it if verbose is True.
    """
    def log(*msg: str) -> None:
        if verbose:
            print(*msg)

    return log


def format_time(seconds: float) -> str:
    """
    Converts a given time in seconds into a readable string format:
    days, hours, minutes, seconds, and milliseconds.

    Args:
        seconds (float): The time duration in seconds to format.

    Returns:
        str: The formatted time string (e.g., '1D 2h 3m 4s 500ms').
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    secondsf = int(seconds)
    millis = int((seconds - secondsf) * 1000)

    time_parts = []
    if days > 0:
        time_parts.append(f"{int(days)}D")
    if hours > 0:
        time_parts.append(f"{int(hours)}h")
    if minutes > 0:
        time_parts.append(f"{int(minutes)}m")
    if secondsf > 0:
        time_parts.append(f"{secondsf}s")
    if millis > 0:
        time_parts.append(f"{millis}ms")

    return ' '.join(time_parts) if time_parts else '0ms'


def rand_slice(data: np.ndarray, slice_len: int = 25) -> np.ndarray:
    """
    Returns a random slice of musical features of size `slice_len + 1`.
    Args:
        data (np.ndarray): The full dataset of musical features.
        slice_len (int): The length of the slice to extract.

    Returns:
        np.ndarray: A random slice of the data.
    """
    d_len = len(data)
    s_idx = random.randint(0, d_len - slice_len - 1)
    e_idx = s_idx + slice_len + 1
    return data[s_idx:e_idx]


def seq_to_tensor(seq: np.ndarray) -> torch.Tensor:
    """
    Convert a sequence of musical features to a PyTorch tensor.
    Args:
        seq (np.ndarray): The sequence of musical features.

    Returns:
        torch.Tensor: A tensor of the same sequence.
    """
    return torch.tensor(seq, dtype=torch.float32)


def train_slice(
        data: np.ndarray,
        slice_len: int = 25) -> Tuple[Variable, Variable]:
    """
    Creates a random training set (input sequence and target sequence).
    Args:
        data (np.ndarray): The full dataset of musical features.
        slice_len (int): The length of the input sequence.

    Returns:
        Tuple[Variable, Variable]: The input sequence and target sequence.
    """
    slice_i = rand_slice(data, slice_len=slice_len)
    seq = seq_to_tensor(slice_i[:-1])
    target = seq_to_tensor(slice_i[1:])
    return Variable(seq), Variable(target)


def train_batch(
        data: np.ndarray,
        b_size: int = 100,
        slice_len: int = 25) -> Tuple[Variable, Variable]:
    """
    Creates a batch of training data.
    Args:
        data (np.ndarray): The full dataset of musical features.
        b_size (int): Batch size.
        slice_len (int): The length of each input sequence.

    Returns:
        Tuple[Variable, Variable]: Batch of input sequences and target
        sequences.
    """
    batch_seq = torch.zeros(b_size, slice_len, data.shape[1])
    batch_target = torch.zeros(b_size, slice_len, data.shape[1])

    for idx in range(b_size):
        seq, target = train_slice(data, slice_len=slice_len)
        batch_seq[idx] = seq.data
        batch_target[idx] = target.data

    return Variable(batch_seq), Variable(batch_target)


def song_to_seq_target(
        song: np.ndarray) -> Tuple[Variable, Variable]:
    """
    Given a song, return a sequence/target as a variable.
    Args:
        song (np.ndarray): The song as an array of musical features.

    Returns:
        Tuple[Variable, Variable]: The input sequence and target sequence.
    """
    a_slice = rand_slice(song)
    seq = seq_to_tensor(a_slice[:-1])
    target = seq_to_tensor(a_slice[1:])
    assert len(seq) == len(target), 'SEQ AND TARGET MISMATCH'
    return Variable(seq), Variable(target)


def save_checkpoint(state: Dict[str, Any], filename: str = 'checkpoint.pth') -> None:
    """
    Save the model checkpoint to a file.

    Args:
        state (dict): State dictionary containing model, optimizer, and other
        parameters.
        filename (str): Name of the checkpoint file.
    """
    print(f"==> Saving checkpoint to {filename}...")
    torch.save(state, filename)


def load_checkpoint(
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Load a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state
        into.

    Returns:
        dict: A dictionary containing model, optimizer, losses, and other
        parameters.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

    print(f"==> Loading checkpoint from {filename}...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
