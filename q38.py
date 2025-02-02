import numpy as np

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

def hermite_interpolation(x_values, y_values, x):
    n = len(x_values)
    slopes = [(y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i]) for i in range(n-1)]
    
    # Using closest two points for interpolation
    x1, x2 = x_values[1], x_values[2]
    y1, y2 = y_values[1], y_values[2]
    dy1, dy2 = slopes[0], slopes[1] 
    
    h = x2 - x1
    t = (x - x1) / h
    
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)
    
    return h00 * y1 + h10 * h * dy1 + h01 * y2 + h11 * h * dy2

# נתונים לנקודות x ו-y
x_values = np.array([6.5, 6.7, 7.0, 8.0])
y_values = np.array([2.14451, 2.35585, 2.74748, 5.67127])

x_target = 6.9

  # חישוב לפי שיטת אינטרפולציה ליניארית
linear_result = linear_interpolation(6.7, 2.35585, 7.0, 2.74748, x_target)
print(f"Linear Interpolation Result: {linear_result}")

# חישוב לפי שיטת הרמיט מדרגה ראשונה
hermite_result = hermite_interpolation(x_values, y_values, x_target)
print(f"Hermite Interpolation Result: {hermite_result}")
