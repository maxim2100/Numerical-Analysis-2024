import math

def formula_1(L):
    return 4.86 + 0.018 * L

def formula_2(L):
    return L / 3000

def formula_3(L, A0=0.0047, A1=0.0023, A2=0.000043):
    return A0 + A1 * L + A2 * (L * math.log(L)) if L > 0 else 0

def formula_4(L):
    return 4 + (0.0015 * (L ** 4.2)) / L if L > 0 else 4

def formula_5(L):
    return 0.069 * L + 0.00156 * (L ** 2) + 0.00000047 * (L ** 3)

data = {
    3: int(1.15063171 * 2500),
    10: int(0.07168566 * 7500),
    11: int(1.385 * 1000),
    21: int(39.508 * 200),
    25: int(2.1 * 1500),
    38: int(2.61 * 350)
}

for question, answer in data.items():
    print("\n--------------------------\nQuestion", question)
    print(f"formula_1: {formula_1(answer)} defects")
    print(f"formula_2: {formula_2(answer)} defects")
    print(f"formula_3: {formula_3(answer)} defects")
    print(f"formula_4: {formula_4(answer)} defects")
    print(f"formula_5: {formula_5(answer)} defects")
