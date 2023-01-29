# DeciderInfo formats

## For DFA half-proofs (ID 10):

    direction:          `uint8_t` (0 if the automaton scans left-to-right, 1 if it scans right-to-left)
    dfa_transitions     `uint8_t[dfa_states][2]` where `dfa_states = (InfoLength-1)/2`

## For NFA proofs (ID unassigned):

    direction:          `uint8_t` (0 if the automaton scans left-to-right, 1 if it scans right-to-left)
    n_states:           `uint8_t`
    for each symbol in 0, 1, A, B, C, D, E
      transitions:      n x n matrix of bits, row-major/little-endian. i.e., there are n ceil(n/8)-bit integers giving the matrix rows, and the ith bit is set if the row's ith entry is set.
    accept:             packed n x 1 vector of bits (little-endian)
    steady_state:       packed 1 x n vector of bits (little-endian)

A transition matrix has the row i, col j entry set if there's a transition from state i to state j.

## For DFA+NFA proofs (ID 11):

    direction:          `uint8_t` (0 if the automaton scans left-to-right, 1 if it scans right-to-left)
    dfa_states:         `uint8_t`
    nfa_states:         `uint8_t`
    dfa_transitions     `uint8_t[dfa_states][2]`
    for each symbol in 0, 1
      nfa_transitions:  n x n matrix of bits, row-major/little-endian. i.e., there are n ceil(n/8)-bit integers giving the matrix rows, and the ith bit is set if the row's ith entry is set.
    accept:             packed n x 1 vector of bits (little-endian)

There must be at least `5*dfa_states+1` NFA states.
For `0 <= q < dfa_states`, `0 <= f < 5`, state `5*q+f` is designated as the result of an `f`-symbol transition from DFA state `q`.
State `5*dfa_states` is reserved for ⊥ (potential halt detected).

# Script

The companion Python script is able to produce these records from a Seed Database and a .dvf file that just contains the DFA side.
These formats are by now quite official, also used in [Tony's repo](https://github.com/TonyGuil/bbchallenge/blob/main/README).
(I believe NFAs can be transformed to ONLY scan left-to-right, and still be checkable, but only if we define transitions for 0, 1, A0, A1, ..., E0, E1 and then write up the procedure in the paper.
 I'd rather not, even though I find the merged-symbol version aesthetically superior.)

The missing features I consider most interesting / most likely to be added:

- A text (probably JSON) format which is much simpler for high-level languages to ingest,
- A more granular CLI that can also sort/deduplicate proofs by TM seed,
- An easy way to export an index file of TMs with verified proofs.
