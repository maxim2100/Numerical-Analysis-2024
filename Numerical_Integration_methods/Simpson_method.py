import math

def simpsons_rule(f, a, b, n):
    """
    חישוב אינטגרל מסוים באמצעות שיטת סימפסון.

    פרמטרים:
    f (function): פונקציה הניתנת לאינטגרציה.
    a (float): גבול תחתון של האינטגרל.
    b (float): גבול עליון של האינטגרל.
    n (int): מספר תתי-המקטעים (חייב להיות זוגי).

    החזרת ערך:
    float: קירוב לערך האינטגרל המסוים של הפונקציה בטווח [a, b].
    """
    if n % 2 != 0:
        n += 1  # הבטחת n זוגי

    h = (b - a) / n
    integral = f(a) + f(b)  # תרומת נקודות הקצה

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x_i)
        else:
            integral += 4 * f(x_i)

    return integral * (h / 3)

def f(x):
    denominator = (2 * x) ** 3 + 5 * x ** 2 - 6
    if denominator == 0:
        return 0  # מניעת חלוקה באפס
    return math.sin(2 * math.e ** (-2 * x)) / denominator

if __name__ == '__main__':
    a, b = -0.5, 0.5
    n = 100  # מספר חיתוכים (זוגי)
    
    result = simpsons_rule(f, a, b, n)
    print(f"Approximate integral using Simpson's rule: {result}")
