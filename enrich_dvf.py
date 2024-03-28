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


class BouncerType(Enum): Unilateral, Bilateral, Translated = range(1, 4)

@dataclass
class TapeDescriptor:
    ''' Wall[0] Repeater[0] ... Wall[nPartitions-1] Repeater[nPartitions-1]  Wall[nPartitions]
    For each partition, the number of repetitions of each Repeater remains unchanged throughout the Cycle, and is found in the RepeaterCount array in the VerificationInfo '''
    state: int
    tape_head_wall: int
    tape_head_offset: int
    wall: list[list[int]]
    repeater: list[list[int]]

    @classmethod
    def read_from(cls, view, n_partitions):
        state, tape_head_wall = view[:2]
        tape_head_offset, view = int.from_bytes(view[2:4], 'big', signed=True), view[4:]
        wall, repeater = arrays = [[None]*(n_partitions+1), [None]*n_partitions]
        for arr in arrays:
            for i in range(len(arr)):
                l = int.from_bytes(view[:2], 'big')
                arr[i] = list(view[2:2+l])
                view = view[2+l:]
        return cls(state, tape_head_wall, tape_head_offset, wall, repeater), view

    def to_bytes(self):
        out = bytearray((self.state, self.tape_head_wall))
        out.extend(self.tape_head_offset.to_bytes(2, 'big', signed=True))
        for arr in self.wall + self.repeater:
            out.extend(len(arr).to_bytes(2, 'big'))
            out.extend(arr)
        return out

@dataclass
class Segment:
    state: int
    tape_head: int
    tape: list[int]

    @classmethod
    def read_from(cls, view):
        state = view[0]
        tape_head = int.from_bytes(view[1:3], 'big', signed=True)
        l = int.from_bytes(view[3:5], 'big')
        return cls(state, tape_head, list(view[5:5+l])), view[5+l:]

    def to_bytes(self):
        out = bytearray((self.state,))
        out.extend(self.tape_head.to_bytes(2, 'big', signed=True))
        out.extend(len(self.tape).to_bytes(2, 'big'))
        out.extend(self.tape)
        return out

@dataclass
class Transition:
    n_steps: int
    initial: Segment
    final: Segment

    @classmethod
    def read_from(cls, view):
        n_steps, view = int.from_bytes(view[:2], 'big'), view[2:]
        initial, view = Segment.read_from(view)
        final, view = Segment.read_from(view)
        return cls(n_steps, initial, final), view

    def to_bytes(self):
        out = bytearray()
        out.extend(self.n_steps.to_bytes(2, 'big'))
        out.extend(self.initial.to_bytes())
        out.extend(self.final.to_bytes())
        return out

@dataclass
class RunDescriptor:
    partition: int
    repeater_transition: Transition
    td0: TapeDescriptor
    transition: Transition
    td1: TapeDescriptor

    @classmethod
    def read_from(cls, view, n_partitions):
        partition, view = view[0], view[1:]
        repeater_transition, view = Transition.read_from(view)
        td0, view = TapeDescriptor.read_from(view, n_partitions)
        transition, view = Transition.read_from(view)
        td1, view = TapeDescriptor.read_from(view, n_partitions)
        return cls(partition, repeater_transition, td0, transition, td1), view

    def to_bytes(self):
        out = bytearray((self.partition,))
        out.extend(self.repeater_transition.to_bytes())
        out.extend(self.td0.to_bytes())
        out.extend(self.transition.to_bytes())
        out.extend(self.td1.to_bytes())
        return out

@dataclass
class DeciderInfoBouncer(DeciderInfo):
    TYPE_ID = 6

    bouncer_type: BouncerType
    n_partitions: int
    n_runs: int
    initial_steps: int
    initial_leftmost: int
    initial_rightmost: int
    final_steps: int
    final_leftmost: int
    final_rightmost: int
    repeater_count: list[int]
    initial_tape: TapeDescriptor
    run_list: list[RunDescriptor]

    @classmethod
    def from_bytes(cls, info_bytes):
        info_bytes = memoryview(info_bytes)
        bouncer_type = BouncerType(info_bytes[0])
        n_partitions = info_bytes[1]
        n_runs = int.from_bytes(info_bytes[2:4], 'big')
        initial_steps = int.from_bytes(info_bytes[4:8], 'big')
        initial_leftmost = int.from_bytes(info_bytes[8:12], 'big', signed=True)
        initial_rightmost = int.from_bytes(info_bytes[12:16], 'big', signed=True)
        final_steps = int.from_bytes(info_bytes[16:20], 'big')
        final_leftmost = int.from_bytes(info_bytes[20:24], 'big', signed=True)
        final_rightmost = int.from_bytes(info_bytes[24:28], 'big', signed=True)
        repeater_count = [int.from_bytes(info_bytes[28+2*i:30+2*i], 'big') for i in range(n_partitions)]
        initial_tape, info_bytes = TapeDescriptor.read_from(info_bytes[28+2*n_partitions:], n_partitions)
        run_list = []
        for _ in range(n_runs):
            run, info_bytes = RunDescriptor.read_from(info_bytes, n_partitions)
            run_list.append(run)
        return cls(bouncer_type, n_partitions, n_runs, initial_steps, initial_leftmost, initial_rightmost, final_steps, final_leftmost, final_rightmost, repeater_count, initial_tape, run_list)

    def to_bytes(self):
        out = bytearray((self.bouncer_type.value, self.n_partitions))
        out.extend(self.n_runs.to_bytes(2, 'big'))
        out.extend(self.initial_steps.to_bytes(4, 'big'))
        out.extend(self.initial_leftmost.to_bytes(4, 'big', signed=True))
        out.extend(self.initial_rightmost.to_bytes(4, 'big', signed=True))
        out.extend(self.final_steps.to_bytes(4, 'big'))
        out.extend(self.final_leftmost.to_bytes(4, 'big', signed=True))
        out.extend(self.final_rightmost.to_bytes(4, 'big', signed=True))
        for rc in self.repeater_count:
            out.extend(rc.to_bytes(2, 'big'))
        out.extend(self.initial_tape.to_bytes())
        for run in self.run_list:
            out.extend(run.to_bytes())
        return out

@dataclass
class NewRunDescriptor:
    partition: int
    repeater_transition: Transition
    transition: Transition

    @classmethod
    def read_from(cls, view, n_partitions):
        partition, view = view[0], view[1:]
        repeater_transition, view = Transition.read_from(view)
        transition, view = Transition.read_from(view)
        return cls(partition, repeater_transition, transition), view

    def to_bytes(self):
        out = bytearray((self.partition,))
        out.extend(self.repeater_transition.to_bytes())
        out.extend(self.transition.to_bytes())
        return out


@dataclass
class DeciderInfoNewBouncer(DeciderInfo):
    TYPE_ID = 7

    bouncer_type: BouncerType
    n_partitions: int
    n_runs: int
    initial_steps: int
    initial_leftmost: int
    initial_rightmost: int
    final_steps: int
    final_leftmost: int
    final_rightmost: int
    repeater_count: list[int]
    initial_tape: TapeDescriptor
    run_list: list[NewRunDescriptor]
    final_adjustment: int
    final_tape: TapeDescriptor

    @classmethod
    def from_bytes(cls, info_bytes):
        info_bytes = memoryview(info_bytes)
        bouncer_type = BouncerType(info_bytes[0])
        n_partitions = info_bytes[1]
        n_runs = int.from_bytes(info_bytes[2:4], 'big')
        initial_steps = int.from_bytes(info_bytes[4:8], 'big')
        initial_leftmost = int.from_bytes(info_bytes[8:12], 'big', signed=True)
        initial_rightmost = int.from_bytes(info_bytes[12:16], 'big', signed=True)
        final_steps = int.from_bytes(info_bytes[16:20], 'big')
        final_leftmost = int.from_bytes(info_bytes[20:24], 'big', signed=True)
        final_rightmost = int.from_bytes(info_bytes[24:28], 'big', signed=True)
        repeater_count = [int.from_bytes(info_bytes[28+2*i:30+2*i], 'big') for i in range(n_partitions)]
        initial_tape, info_bytes = TapeDescriptor.read_from(info_bytes[28+2*n_partitions:], n_partitions)
        run_list = []
        for _ in range(n_runs):
            run, info_bytes = NewRunDescriptor.read_from(info_bytes, n_partitions)
            run_list.append(run)
        final_adjustment = int.from_bytes(info_bytes[:2], 'big')
        final_tape, info_bytes = TapeDescriptor.read_from(info_bytes[2:], n_partitions)
        return cls(bouncer_type, n_partitions, n_runs, initial_steps, initial_leftmost, initial_rightmost, final_steps, final_leftmost, final_rightmost, repeater_count, initial_tape, run_list, final_adjustment, final_tape)

    def to_bytes(self):
        out = bytearray((self.bouncer_type.value, self.n_partitions))
        out.extend(self.n_runs.to_bytes(2, 'big'))
        out.extend(self.initial_steps.to_bytes(4, 'big'))
        out.extend(self.initial_leftmost.to_bytes(4, 'big', signed=True))
        out.extend(self.initial_rightmost.to_bytes(4, 'big', signed=True))
        out.extend(self.final_steps.to_bytes(4, 'big'))
        out.extend(self.final_leftmost.to_bytes(4, 'big', signed=True))
        out.extend(self.final_rightmost.to_bytes(4, 'big', signed=True))
        for rc in self.repeater_count:
            out.extend(rc.to_bytes(2, 'big'))
        out.extend(self.initial_tape.to_bytes())
        for run in self.run_list:
            out.extend(run.to_bytes())
        out.extend(self.final_adjustment.to_bytes(2, 'big'))
        out.extend(self.final_tape.to_bytes())
        return out


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
