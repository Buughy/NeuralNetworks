import pathlib
import re

def parse_line(line: str) -> tuple[list[float], float]:
    terms, result = line.split('=')
    result = float(result.strip())
    terms = terms.replace(' ', '')

    terms = [c for c, l in re.findall(r'([-]?\d*)([a-zA-Z])', terms)]

    coefficients = [0, 0, 0]
    index = 0

    for term in terms:
        if (term == '-'):
            coeff = -1
        elif (term == ''):
            coeff = 1
        else:
            coeff = float(term)
        coefficients[index] = coeff
        index += 1

    return coefficients, result

def load_from_file(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []

    with path.open() as file:
        for line in file:
            coefficients, constant = parse_line(line.strip())
            A.append(coefficients)
            B.append(constant)

    return A, B

