import math
import numpy as np

def max_steps(a, b, err):
    """ מחשב את מספר הצעדים המקסימלי הנדרש להגעה לדיוק המבוקש """
    return int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))

def bisection_method(f, a, b, tol=1e-6):
    """
    מבצע את שיטת החצייה (Bisection Method) כדי למצוא שורש של פונקציה f בטווח [a,b]
    :param f: פונקציה מתמטית רציפה בטווח [a,b]
    :param a: גבול תחתון
    :param b: גבול עליון
    :param tol: דיוק רצוי (ברירת מחדל 1e-6)
    :return: שורש מקורב של הפונקציה f
    """
    fa, fb = f(a), f(b)
    
    if np.sign(fa) == np.sign(fb):
        raise ValueError(f"No root found in interval [{a}, {b}]")

    steps = max_steps(a, b, tol)  

    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    for k in range(steps):
        c = (a + b) / 2  
        fc = f(c)

        print("{:<10} {:<15.9f} {:<15.9f} {:<15.9f} {:<15.9f} {:<15.9f} {:<15.9f}".format(k, a, b, fa, fb, c, fc))

        if abs(fc) < tol:  
            print(f"\nThe equation f(x) has an approximate root at x = {c:<15.9f}")
            return c

        if np.sign(fc) == np.sign(fa):  
            a, fa = c, fc
        else:  
            b, fb = c, fc

    print(f"\nMaximum iterations reached. Approximate root at x = {c:<15.9f}")
    return c


def find_root_intervals(f, x_start, x_end, step=0.1):
    """
    מוצא טווחים שבהם יש שורשים על ידי בדיקת שינויי סימן של הפונקציה f(x)
    :param f: פונקציה מתמטית רציפה
    :param x_start: התחלה של הסריקה
    :param x_end: סוף הסריקה
    :param step: גודל הקפיצה (ברירת מחדל 0.1)
    :return: רשימה של זוגות (a, b) שמכילים שורשים
    """
    root_intervals = []
    x_values = np.arange(x_start, x_end, step)

    for i in range(len(x_values) - 1):
        x0, x1 = x_values[i], x_values[i + 1]
        f0, f1 = f(x0), f(x1)

        if np.sign(f0) != np.sign(f1):  
            root_intervals.append((x0, x1))

    return root_intervals

