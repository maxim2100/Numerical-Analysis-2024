import math

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))  # חישוב התרומה של נקודות הקצה

    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)

    return integral * h

def f(x):
    denominator = (2 * x) ** 3 + 5 * x ** 2 - 6
    if denominator == 0:
        return 0  # מניעת חלוקה באפס
    return math.sin(2 * math.e ** (-2 * x)) / denominator

if __name__ == '__main__':
    a, b = -0.5, 0.5
    n = 100  # מספר חיתוכים - ניתן להגדיל לדיוק גבוה יותר
    result = trapezoidal_rule(f, a, b, n)
    print("Approximate integral:", result)
