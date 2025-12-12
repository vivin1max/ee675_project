import numpy as np
from typing import List, Tuple, Union
from edit_distance import SequenceMatcher


def compute_edit_distance(ref: Union[List, np.ndarray], hyp: Union[List, np.ndarray]) -> int:

    if isinstance(ref, np.ndarray):
        ref = ref.tolist()
    if isinstance(hyp, np.ndarray):
        hyp = hyp.tolist()
    
    matcher = SequenceMatcher(a=ref, b=hyp)
    return matcher.distance()


def compute_cer(
    ref: Union[List, np.ndarray],
    hyp: Union[List, np.ndarray]
) -> Tuple[int, int]:

    edit_dist = compute_edit_distance(ref, hyp)
    ref_len = len(ref)
    return edit_dist, ref_len


def compute_cer_batch(
    refs: List[Union[List, np.ndarray]],
    hyps: List[Union[List, np.ndarray]]
) -> float:

    total_edit_distance = 0
    total_length = 0
    
    for ref, hyp in zip(refs, hyps):
        edit_dist, ref_len = compute_cer(ref, hyp)
        total_edit_distance += edit_dist
        total_length += ref_len
    
    return total_edit_distance / total_length if total_length > 0 else 1.0


def compute_wer(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int]:

    edit_dist = compute_edit_distance(ref_words, hyp_words)
    ref_len = len(ref_words)
    return edit_dist, ref_len


def compute_wer_batch(
    refs: List[List[str]],
    hyps: List[List[str]]
) -> float:

    total_edit_distance = 0
    total_length = 0
    
    for ref, hyp in zip(refs, hyps):
        edit_dist, ref_len = compute_wer(ref, hyp)
        total_edit_distance += edit_dist
        total_length += ref_len
    
    return total_edit_distance / total_length if total_length > 0 else 1.0


def phoneme_sequence_to_words(
    phoneme_seq: Union[List, np.ndarray],
    phoneme_to_word_mapping: dict = None
) -> List[str]:

    if phoneme_to_word_mapping is None:
        return [str(p) for p in phoneme_seq]
    
    words = []
    for phoneme in phoneme_seq:
        if phoneme in phoneme_to_word_mapping:
            words.append(phoneme_to_word_mapping[phoneme])
    
    return words


class MetricsTracker:

    
    def __init__(self):

        self.reset()
    
    def reset(self):

        self.total_edit_distance = 0
        self.total_seq_length = 0
        self.num_sequences = 0
    
    def update(self, ref: Union[List, np.ndarray], hyp: Union[List, np.ndarray]):

        edit_dist, seq_len = compute_cer(ref, hyp)
        self.total_edit_distance += edit_dist
        self.total_seq_length += seq_len
        self.num_sequences += 1
    
    def update_batch(
        self,
        refs: List[Union[List, np.ndarray]],
        hyps: List[Union[List, np.ndarray]]
    ):

        for ref, hyp in zip(refs, hyps):
            self.update(ref, hyp)
    
    def compute(self) -> float:

        if self.total_seq_length == 0:
            return 1.0
        return self.total_edit_distance / self.total_seq_length
    
    def get_stats(self) -> dict:

        return {
            'cer': self.compute(),
            'total_edit_distance': self.total_edit_distance,
            'total_seq_length': self.total_seq_length,
            'num_sequences': self.num_sequences,
            'avg_seq_length': self.total_seq_length / self.num_sequences if self.num_sequences > 0 else 0,
        }


def print_metrics_summary(metrics: dict, title: str = "Metrics"):

    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
