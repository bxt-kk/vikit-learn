from typing import Dict, List
from collections import defaultdict
import json
import os
from datetime import datetime

from torch import Tensor
import matplotlib.pyplot as plt


class Logger:
    DEFAULT_MODES = [
        'train',
        'valid',
        'test',
        'metric',
    ]

    def __init__(
            self,
            name:    str | None=None,
            log_dir: str | None=None,
        ):

        self._name = name or 'TRAIN'
        self.reset()

        self._log_dir = log_dir or os.path.join(
            os.path.abspath(os.path.curdir), 'logs')
        if not os.path.isdir(self._log_dir):
            os.makedirs(self._log_dir, exist_ok=True)
        self._dump_path = os.path.join(self._log_dir, (
            f'{self._name}-LOG-'
            f'{datetime.now():%y%m%d-%H:%M:%S}'
            '.txt'
        ))

    def reset(self):
        self._step = defaultdict(int)
        self._logs = {mode: defaultdict(float)
            for mode in self.DEFAULT_MODES}

    def update(
            self,
            mode:   str,
            *datas: Dict[str, float | Tensor],
        ):

        for data in datas:
            for key, raw in data.items():
                if isinstance(raw, Tensor):
                    value = raw.item()
                else:
                    value = raw
                self._logs[mode][key] += value
        self._step[mode] += 1

    def compute(
            self,
            mode: str,
            mean: bool=True,
        ) -> Dict[str, float] | None:

        if self._step[mode] < 1: return dict()
        step = self._step[mode] if mean else 1
        return {
            k: round(v / step, 5)
            for k, v in self._logs[mode].items()}

    def dumps(self, *modes:str) -> str:
        modes = modes or self.DEFAULT_MODES
        datas = {mode: self.compute(mode) for mode in modes}
        return json.dumps(datas, ensure_ascii=False)

    def dumpf(self, *modes:str) -> str:
        dump_str = self.dumps(*modes)
        with open(self._dump_path, 'a') as f:
            f.write(dump_str + '\n')
        return dump_str

    def plot(self):
        self.plot_from_disk(self._dump_path)

    @classmethod
    def plot_from_disk(cls, filepath:str):
        frames:Dict[str, Dict[str, List]] = dict()
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            datas = json.loads(line)
            for mode, data in datas.items():
                for key, value in data.items():
                    if key not in frames:
                        frames[key] = dict()
                    if mode not in frames[key]:
                        frames[key][mode] = list()
                    frames[key][mode].append(value)

        path = os.path.splitext(filepath)[0]
        for key, frame in frames.items():
            plt.figure()
            plt.title(key)
            plt.xlabel('epoch')
            for mode, values in frame.items():
                plt.plot(range(1, len(values) + 1), values, label=mode)
            plt.legend()
            plt.savefig(f'{path}_{key}.png')

    @classmethod
    def create_by_output(cls, output:str) -> 'Logger':
        basename = os.path.basename(output).upper()
        log_dir = os.path.join(os.path.dirname(output), 'logs')
        return cls(name=basename, log_dir=log_dir)
