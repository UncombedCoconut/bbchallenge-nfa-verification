# Warning: the below format is superseded. A verification file with ID=11 should now conform to [this spec](https://github.com/TonyGuil/bbchallenge/blob/main/FAR/ReadMe.md)

Tentative DeciderInfo format for NFA proofs (ID = 11):

    direction:          uint8_t (0 if the automaton scans left-to-right, 1 if it scans right-to-left)
    n_states:           uint8_t
    for each symbol in 0, 1, A, B, C, D, E
        transitions:    n x n matrix of bits, row-major/little-endian. i.e., there are n ceil(n/8)-bit integers giving the matrix rows, and the ith bit is set if the row's ith entry is set.
    accept:             packed n x 1 vector of bits (little-endian)
    steady_state:       packed 1 x n vector of bits (little-endian)

A transition matrix has the row i, col j entry set if there's a transition from state i to state j.

The companion Python script is able to produce these records from a Seed Database and a .dvf file that just contains the DFA side.
The main doubt about this format is whether the complexity can be avoided.
(I believe NFAs can be transformed to ONLY scan left-to-right, and still be checkable, but only if we define transitions for 0, 1, A0, A1, ..., E0, E1 and then write up the procedure in the paper.
 I'd rather not, even though I find the merged-symbol version aesthetically superior.)

The missing features I consider most interesting / most likely to be added:

- A text (probably JSON) format which is much simpler for high-level languages to ingest,
- A more granular CLI that can also sort/deduplicate proofs by TM seed,
- An easy way to export an index file of TMs with verified proofs.
