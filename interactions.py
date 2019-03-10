import numpy as np
from numpy import int8

# Representing rock, paper, and scissors with simple 8-bit integers.
ROCK, PAPER, SCISSORS = int8(1), int8(2), int8(3)


def rock_paper_scissors_interaction(parameters, microbe_properties, p1, p2):
    pRS, pPR, pSP = parameters["pRS"], parameters["pPR"], parameters["pSP"]
    species = microbe_properties["species"]

    if species[p1] != species[p2]:
        s1, s2 = species[p1], species[p2]

        r = np.random.rand()  # Random float from Uniform[0,1)

        winner = None

        if s1 == ROCK and s2 == SCISSORS:
            winner = p1 if r < pRS else p2
        elif s1 == ROCK and s2 == PAPER:
            winner = p2 if r < pPR else p1
        elif s1 == PAPER and s2 == ROCK:
            winner = p1 if r < pPR else p2
        elif s1 == PAPER and s2 == SCISSORS:
            winner = p2 if r < pSP else p1
        elif s1 == SCISSORS and s2 == ROCK:
            winner = p2 if r < pRS else p1
        elif s1 == SCISSORS and s2 == PAPER:
            winner = p1 if r < pSP else p2

        if winner == p1:
            species[p2] = species[p1]
        elif winner == p2:
            species[p1] = species[p2]


def rock_paper_scissors(N_microbes, pRS=0.5, pPR=0.5, pSP=0.5):
    microbe_properties = {
        "species": np.random.choice([ROCK, PAPER, SCISSORS], N_microbes)
    }

    interaction_parameters = {
        "pRS": pRS,  # Forward probability that rock beats scissors.
        "pPR": pPR,  # Forward probability that paper beats rock.
        "pSP": pSP   # Forward probability that scissors beats paper.
    }

    return rock_paper_scissors_interaction, interaction_parameters, microbe_properties

