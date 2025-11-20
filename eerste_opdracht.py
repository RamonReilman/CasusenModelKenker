#imports
import random
import numpy as np

#Nucleotidenlijst
NUCS = ["A", "C", "G", "T"]

#Kansen van de brighspace
CGI_TRANS = {
    'A': [0.180, 0.274, 0.426, 0.120],
    'C': [0.171, 0.368, 0.274, 0.188],
    'G': [0.161, 0.339, 0.375, 0.125],
    'T': [0.079, 0.355, 0.384, 0.182],
}

NON_CGI_TRANS = {
    'A': [0.300, 0.205, 0.285, 0.210],
    'C': [0.322, 0.298, 0.078, 0.302],
    'G': [0.248, 0.246, 0.298, 0.208],
    'T': [0.177, 0.239, 0.292, 0.292],
}


def pick_next(base, matrix):
    r = random.random()
    cum = 0
    for nuc, p in zip(NUCS, matrix[base]):
        cum += p
        if r < cum:
            return nuc
    return NUCS[-1]

def generate_sequence(length, matrix):
    seq = [random.choice(NUCS)]
    for _ in range(length - 1):
        seq.append(pick_next(seq[-1], matrix))
    return "".join(seq)

def mono_freqs(seq):
    arr = np.zeros(4)
    for s in seq:
        arr[NUCS.index(s)] += 1
    return arr / len(seq)

def di_freqs(seq):
    arr = np.zeros((4, 4))
    for a, b in zip(seq[:-1], seq[1:]):
        arr[NUCS.index(a), NUCS.index(b)] += 1
    return arr / (len(seq) - 1)

def di_ratio(mono, di):
    # expected = P_X * P_Y
    expected = mono.reshape(4,1) * mono.reshape(1,4)
    return di / expected

def print_output(name, seq):
    print(f"{name}\n")
    print("Sequence:")
    print(seq, "\n")

    mono = mono_freqs(seq)
    di = di_freqs(seq)
    ratio = di_ratio(mono, di)

    print("Observed nucleotides (ACGT):")
    print(np.round(mono, 3), "\n")

    print("Observed dinucleotides:")
    print(np.round(di, 3), "\n")

    print("Observed/Expected ratio:")
    print(np.round(ratio, 3), "\n")


# MAIN
random.seed(42)
# input voor de binnen de eilanden
seq_cgi = generate_sequence(300, CGI_TRANS)
print_output("CGI+", seq_cgi)
# input voor de buiten de eilanden
seq_non_cgi = generate_sequence(300, NON_CGI_TRANS)
print_output("CGI-", seq_non_cgi)
