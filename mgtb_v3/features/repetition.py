from __future__ import annotations

from collections import defaultdict

from mgtb_v3.features.confident_loop import confident_loop_delta
from mgtb_v3.types import NgramOccurrence


class NgramTracker:
    def __init__(self, n_min: int, n_max: int, prompt_tokens=None, exclude_prompt: bool = True):
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.exclude_prompt = bool(exclude_prompt)
        self.prompt_tokens = list(prompt_tokens or [])
        self.prompt_len = len(self.prompt_tokens)
        self.prompt_ngrams = self._collect_prompt_ngrams() if exclude_prompt else set()
        self.tokens: list[int] = []
        self.logprobs: list[float] = []
        self.positions: list[int] = []
        self.occurrences_by_ngram: dict[tuple[int, ...], list[NgramOccurrence]] = defaultdict(list)
        self.repeated_occurrences: list[NgramOccurrence] = []
        self.loop_deltas: list[tuple[NgramOccurrence, float]] = []

    def _collect_prompt_ngrams(self) -> set[tuple[int, ...]]:
        found = set()
        for n in range(self.n_min, self.n_max + 1):
            for i in range(0, max(0, len(self.prompt_tokens) - n + 1)):
                found.add(tuple(self.prompt_tokens[i : i + n]))
        return found

    def update(self, tokens, logprobs, current_pos: int) -> list[NgramOccurrence]:
        start_idx = len(self.tokens)
        for offset, token in enumerate(tokens):
            self.tokens.append(int(token))
            self.logprobs.append(float(logprobs[offset]))
            self.positions.append(int(current_pos + offset))

        new_repeats: list[NgramOccurrence] = []
        for end_idx in range(start_idx, len(self.tokens)):
            for n in range(self.n_min, self.n_max + 1):
                start = end_idx - n + 1
                if start < 0:
                    continue
                ngram = tuple(self.tokens[start : end_idx + 1])
                if self.exclude_prompt and ngram in self.prompt_ngrams:
                    continue
                mean_logprob = sum(self.logprobs[start : end_idx + 1]) / n
                occurrence = NgramOccurrence(
                    ngram=ngram,
                    start_pos=self.positions[start],
                    end_pos=self.positions[end_idx] + 1,
                    mean_logprob=float(mean_logprob),
                )
                previous = self.occurrences_by_ngram[ngram]
                if previous:
                    new_repeats.append(occurrence)
                    self.repeated_occurrences.append(occurrence)
                    delta = confident_loop_delta(mean_logprob, [item.mean_logprob for item in previous])
                    self.loop_deltas.append((occurrence, delta))
                previous.append(occurrence)
        return new_repeats

    def repetition_rate(self, start_pos: int, end_pos: int) -> float:
        denominator = 0
        for n in range(self.n_min, self.n_max + 1):
            for start in range(len(self.positions) - n + 1):
                occ_start = self.positions[start]
                occ_end = self.positions[start + n - 1] + 1
                if start_pos <= occ_start and occ_end <= end_pos:
                    ngram = tuple(self.tokens[start : start + n])
                    if self.exclude_prompt and ngram in self.prompt_ngrams:
                        continue
                    denominator += 1
        if denominator == 0:
            return 0.0
        numerator = sum(1 for occ in self.repeated_occurrences if start_pos <= occ.start_pos and occ.end_pos <= end_pos)
        return float(numerator / denominator)

    def confident_loop_score(self, start_pos: int, end_pos: int) -> float:
        values = [delta for occ, delta in self.loop_deltas if start_pos <= occ.start_pos and occ.end_pos <= end_pos]
        return float(max(values)) if values else 0.0

    def faulty_ngrams(self, start_pos: int, end_pos: int, top_k: int = 10) -> list[tuple[int, ...]]:
        scored = [
            (delta, occ.ngram)
            for occ, delta in self.loop_deltas
            if start_pos <= occ.start_pos and occ.end_pos <= end_pos
        ]
        scored.extend(
            (0.0, occ.ngram)
            for occ in self.repeated_occurrences
            if start_pos <= occ.start_pos and occ.end_pos <= end_pos
        )
        scored.sort(reverse=True, key=lambda item: item[0])
        seen = set()
        output = []
        for _, ngram in scored:
            if ngram in seen:
                continue
            seen.add(ngram)
            output.append(ngram)
            if len(output) >= top_k:
                break
        return output

    def truncate(self, pos: int) -> None:
        kept = [(t, lp, p) for t, lp, p in zip(self.tokens, self.logprobs, self.positions) if p < pos]
        self.tokens = []
        self.logprobs = []
        self.positions = []
        self.occurrences_by_ngram = defaultdict(list)
        self.repeated_occurrences = []
        self.loop_deltas = []
        if kept:
            tokens, logprobs, positions = zip(*kept)
            self.update(tokens, logprobs, int(positions[0]))
