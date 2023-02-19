#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT

from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import numpy as np
import struct


TM_STATES = 5
class Direction(Enum): R, L = range(2)
class CheckResult(Enum): PASS, FAIL, SKIP = range(3)


@dataclass
class DvfRecord:
    DISPATCH = {}

    seed: int
    info: 'DeciderInfo'

    @classmethod
    def read_from(cls, file):
        try:
            seed, type_id, length = struct.unpack('>LLL', file.read(12))
            info_bytes = file.read(length)
            info = cls.DISPATCH[type_id].from_bytes(info_bytes)
        except KeyError:
           info = DeciderInfoUnknown(type_id, info_bytes)
        except struct.error:
            return None
        return cls(seed, info)

    def write_to(self, file):
        info_bytes = self.info.to_bytes()
        file.writelines((struct.pack('>LLL', self.seed, self.info.TYPE_ID, len(info_bytes)), info_bytes))


class DeciderInfo:
    def check(self, tm):
        return CheckResult.SKIP

    def __init_subclass__(cls):
        if hasattr(cls, 'TYPE_ID'):
            DvfRecord.DISPATCH[cls.TYPE_ID] = cls


@dataclass
class DeciderInfoUnknown(DeciderInfo):
    TYPE_ID: int
    info: bytes

    def to_bytes(self):
        return self.info


@dataclass
class DeciderInfoDFA(DeciderInfo):
    TYPE_ID = 10

    direction: Direction
    transitions: list[tuple[int, int]]

    @classmethod
    def from_bytes(cls, info_bytes):
        return cls(Direction(info_bytes[0]), list(zip(info_bytes[1::2], info_bytes[2::2])))

    def to_bytes(self):
        out = bytearray((self.direction.value,))
        out.extend(val for row in self.transitions for val in row)
        return out


@dataclass
class DeciderInfoNFA(DeciderInfo):
    TYPE_ID = 0xff # DO NOT USE

    direction: Direction
    n_states: int
    transitions: list[np.ndarray]   # n_states x n_states each for symbols 0, 1, then each TM state
    accept: np.ndarray              # n_states x 1
    steady_state: np.ndarray        # 1 x n_states

    @classmethod
    def from_bytes(cls, info_bytes):
        dir_code, n = info_bytes[:2]
        offset = 2
        vec_len= (n+7) // 8
        transitions = []
        for _ in range(2 + TM_STATES):
            transitions.append( np.unpackbits(np.frombuffer(info_bytes, np.uint8, offset=offset, count=n*vec_len).reshape((n, vec_len)), 1, n, 'little').astype(bool) )
            offset += n * vec_len
        accept = np.unpackbits(np.frombuffer(info_bytes, np.uint8, offset=offset, count=vec_len).reshape((vec_len, 1)), 0, n, 'little').astype(bool)
        offset += vec_len
        steady_state = np.unpackbits(np.frombuffer(info_bytes, np.uint8, offset=offset, count=vec_len).reshape((1, vec_len)), 1, n, 'little').astype(bool)
        return cls(Direction(dir_code), n, transitions, accept, steady_state)

    def to_bytes(self):
        out = bytearray((self.direction.value, self.n_states))
        for mat in self.transitions:
            out.extend(np.packbits(mat, 1, 'little').tobytes())
        out.extend(np.packbits(self.accept, 0, 'little').tobytes())
        out.extend(np.packbits(self.steady_state, 1, 'little').tobytes())
        return out

    def check(self, tm):
        T = self.transitions
        q0 = np.zeros((1, self.n_states), bool)
        q0[0, 0] = True
        if not np.array_equal(q0 @ T[0], q0):
            return CheckResult.FAIL  # (1) leading zeros not ignored
        if not np.array_equal(T[0] @ self.accept, self.accept):
            return CheckResult.FAIL  # (2) trailing zeros not ignored
        if (q0 @ T[2+0] @ self.accept).any():
            return CheckResult.FAIL  # (3) initial configuration not rejected
        if not (self.steady_state @ self.accept).any():
            return CheckResult.FAIL  # (4) steady state not accepted
        if (not (self.steady_state >= self.steady_state @ T[0]).all()
         or not (self.steady_state >= self.steady_state @ T[1]).all()):
            return CheckResult.FAIL  # (5) "steady" state isn't

        # Unfortunately, equations (6)/(7) are nasty for a general NFA proof. We need to form a basis for the image of q0 under all compositions of 0/1 transitions.
        # We shall construct a matrix B whose rows form this basis. (Repeatedly replace B's rows with their images under T[0], T[1]. As T[0] fixes q0, the new rows include the old.)
        B, old_dimension = q0, 0
        while old_dimension < B.shape[0]:
            old_dimension = B.shape[0]
            B = np.unique(np.block([[B @ T[b]] for b in range(2)]), axis=0)

        if not B.any(axis=1).all():
            return CheckResult.FAIL  # (6) q_0 T_u was the 0 vector, for some word u.
        for f in range(TM_STATES):
            for r in range(2):
                w, s, t  = tm[3*(2*f+r):3*(2*f+r+1)]
                t -= 1 # Compensate for seed DB offset.
                if t == -1:
                    if not ((B @ T[2+f] @ T[r]).min(axis=0) >= self.steady_state).all():
                        return CheckResult.FAIL  # (7) Fails to transition immediate halts to the steady state.
                elif s == self.direction.value:  # TM transition from [fr] to [wt]
                    if not (T[2+f] @ T[r] >= T[w] @ T[2+t]).all():
                        return CheckResult.FAIL  # (8) Not closed under time-reversed TM transitions moving in the scan direction.
                else:  # TM transition from [bfr] to [tbw]
                    for b in range(2):
                        if not (T[b] @ T[2+f] @ T[r] >= T[2+t] @ T[b] @ T[w]).all():
                            return CheckResult.FAIL  # (9) Not closed under time-reversed TM transitions moving in the scan direction.
        return CheckResult.PASS


def read_dvf(path):
    with open(path, 'rb') as dvf:
        dvf.read(4)  # Skip record count
        while (record := DvfRecord.read_from(dvf)):
            yield record


def compute_nfa_info(tm, dfa_info):
    dir_code = dfa_info.direction.value
    nD = len(dfa_info.transitions)
    nN = TM_STATES * nD + 1
    n = nD + nN
    transitions = [np.zeros((n, n), bool) for _ in range(2 + TM_STATES)]
    halt = np.zeros((1, n), bool)

    # Define the NFA as in paper section "Search algorithm: direct", with initially empty "R_b" and "a".
    q_halt = n-1
    for q, destinations in enumerate(dfa_info.transitions):
        for b, delta_q_b in enumerate(destinations):
            transitions[b][q, delta_q_b] = True
    for q_tm in range(TM_STATES):
        for q_dfa in range(nD):
            transitions[2+q_tm][q_dfa, nD + q_dfa*TM_STATES+q_tm] = True
    halt[0, q_halt] = True
    # Apply equation (5') -- halted transitions to halted.
    transitions[0][q_halt, q_halt] = transitions[1][q_halt, q_halt] = True
    # Seek a fixed point of applying (7')-(9') -- closure under time-reversed TM transitions.
    converged = False
    while not converged:
        converged = True
        for r in range(2):
            t_r, old = transitions[r], transitions[r].copy()
            for f in range(TM_STATES):
                w, s, t  = tm[3*(2*f+r):3*(2*f+r+1)]
                t -= 1 # Compensate for seed DB offset.
                if t == -1:
                    for q_dfa in range(nD):
                        t_r[nD + q_dfa*TM_STATES+f, q_halt] = True
                elif s == dir_code:  # TM transition from [fr] to [wt]
                    t_r |= transitions[2+f].T @ transitions[w] @ transitions[2+t]
                else:  # TM transition from [bfr] to [tbw]
                    for b in range(2):
                        t_r |= (transitions[b] @ transitions[2+f]).T @ transitions[2+t] @ transitions[b] @ transitions[w]
            converged &= np.array_equal(t_r, old)
    # Calculate the accepted-state vector as a fixed point of halt.T
    accept = halt.T
    while True:
        old, accept = accept.copy(), transitions[0] @ accept
        if np.array_equal(old, accept): break

    return DeciderInfoHybrid(dfa_info.direction, nD, nN, dfa_info.transitions, [transitions[0][nD:n, nD:n], transitions[1][nD:, nD:]], accept[nD:, :])


@dataclass
class DeciderInfoHybrid(DeciderInfo):
    TYPE_ID = 11

    direction: Direction
    dfa_states: int
    nfa_states: int
    dfa_transitions: list[tuple[int, int]]
    nfa_transitions: list[np.ndarray]   # n_states x n_states each for symbols 0, 1
    accept: np.ndarray              # n_states x 1

    @classmethod
    def from_bytes(cls, info_bytes):
        dir_code = info_bytes[0]
        nD = int.from_bytes(info_bytes[1:3], 'big')
        nN = int.from_bytes(info_bytes[3:5], 'big')
        offset = 5
        dfa_transitions = list(zip(info_bytes[5:2*nD+5:2], info_bytes[6:2*nD+6:2]))
        offset += 2*nD
        vec_len= (nN+7) // 8
        nfa_transitions = []
        for _ in range(2):
            nfa_transitions.append( np.unpackbits(np.frombuffer(info_bytes, np.uint8, offset=offset, count=nN*vec_len).reshape((nN, vec_len)), 1, nN, 'little').astype(bool) )
            offset += nN * vec_len
        accept = np.unpackbits(np.frombuffer(info_bytes, np.uint8, offset=offset, count=vec_len).reshape((vec_len, 1)), 0, nN, 'little').astype(bool)
        return cls(Direction(dir_code), nD, nN, dfa_transitions, nfa_transitions, accept)

    def to_bytes(self):
        out = bytearray((self.direction.value,))
        out.extend(self.dfa_states.to_bytes(2, 'big'))
        out.extend(self.nfa_states.to_bytes(2, 'big'))
        out.extend(val for row in self.dfa_transitions for val in row)
        for mat in self.nfa_transitions:
            out.extend(np.packbits(mat, 1, 'little').tobytes())
        out.extend(np.packbits(self.accept, 0, 'little').tobytes())
        return out

    def check(self, tm):
        dir_code = self.direction.value
        nD = self.dfa_states
        nN = self.nfa_states
        if nN < TM_STATES * nD + 1 or not all(0 <= ti < nD for t01 in self.dfa_transitions for ti in t01):
            return CheckResult.FAIL
        n = nD + nN
        transitions = [np.zeros((n, n), bool) for _ in range(2 + TM_STATES)]
        halt = np.zeros((1, n), bool)

        # Define the NFA as in paper section "Search algorithm: direct", with initially empty "R_b" and "a".
        q_halt = n-1
        for q, destinations in enumerate(self.dfa_transitions):
            for b, delta_q_b in enumerate(destinations):
                transitions[b][q, delta_q_b] = True
        for q_tm in range(TM_STATES):
            for q_dfa in range(nD):
                transitions[2+q_tm][q_dfa, nD + q_dfa*TM_STATES+q_tm] = True
        halt[0, q_halt] = True
        for b in range(2):
            transitions[b][nD:, nD:] = self.nfa_transitions[b]
        accept = np.zeros((n, 1), bool)
        accept[nD:, :] = self.accept
        return DeciderInfoNFA(self.direction, n, transitions, accept, halt).check(tm)


if __name__ == '__main__':
    ap = ArgumentParser(description='Enrich and check verification data for Finite Automata Reduction.')
    ap.add_argument('-d', '--db', help='Path to DB file', default='../../all_5_states_undecided_machines_with_global_header')
    ap.add_argument('-o', '--output', help='Output DVF path', default='full_nfas.dvf')
    ap.add_argument('dvf', help='Input DVF paths', nargs='*', default=['../output/finite_automata_reduction.dvf'])
    # TODO: Would be nice if we could sort records, dedup proofs for the same seed, and output an index of verified machines.
    args = ap.parse_args()

    with open(args.db, 'rb') as db, open(args.output, 'wb') as output:
        output.write(b'\0\0\0\0')  # Unknown length.
        records = (r for path in args.dvf for r in read_dvf(path))
        stats = dict.fromkeys(CheckResult, 0)

        for record in tqdm(records, unit=' beaver'):
            db.seek(6*TM_STATES * record.seed + 30)
            tm = db.read(6*TM_STATES)
            if isinstance(record.info, DeciderInfoDFA):
                record.info = compute_nfa_info(tm, record.info)
            result = record.info.check(tm)
            stats[result] += 1
            if result != CheckResult.FAIL:
                record.write_to(output)

    with open(args.output, 'r+b') as output:
        output.write(struct.pack('>L', stats[CheckResult.PASS]+stats[CheckResult.SKIP]))

    print('Processing results:', stats)
