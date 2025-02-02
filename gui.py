import tkinter as tk
from tkinter import ttk, messagebox
from src.machine_precision.machine_precision import machine_epsilon
from src.matrices.matrix_operations import MatrixOperations
import numpy as np

class MachinePrecisionFrame(tk.Frame):
    """Frame for machine precision calculations."""
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Machine Precision Calculator").pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.btn_compute = tk.Button(self, text="Compute Machine Precision", command=self.compute)
        self.btn_compute.pack()

    def compute(self):
        eps = machine_epsilon()
        self.result_label.config(text=f"Machine Epsilon: {eps:.16f}")

class InverseMatrixFrame(tk.Frame):
    """
    Frame for entering an n x n matrix, then computing and displaying its inverse.
    """
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # 1) A label + entry to choose matrix order
        self.label_n = tk.Label(self, text="Choose Order Of Matrix", font=("Courier", 10, "bold"))
        self.label_n.pack(pady=(5,0))

        self.entry_n = tk.Entry(self, width=5)
        self.entry_n.pack(pady=(0,10))
        # Start with empty:
        self.entry_n.insert(0, "")

        # 2) Button to build the entry grid
        self.btn_update = tk.Button(self, text="Update Matrix Size", command=self.create_matrix_entries)
        self.btn_update.pack(pady=(0,10))

        # 3) Title label for entering coefficients
        self.label_title = tk.Label(self, text="Enter Coefficients of Matrix:", font=("Courier", 14, "bold"))
        self.label_title.pack()

        # 4) Frame that will hold the matrix's Entry widgets
        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.pack(pady=5)

        self.entries_matrix = []  # 2D list of Entry widgets

        # 5) A bottom frame for the Calculate/Clear buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        self.btn_calculate = tk.Button(button_frame, text="CALCULATE", bg="#4CAF50", fg="white", 
                                       command=self.on_calculate)
        self.btn_calculate.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(button_frame, text="CLEAR", bg="#B71C1C", fg="white",
                                   command=self.on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.text_steps = tk.Text(self, height=20, width=80)
        self.text_steps.pack(pady=10)

        # 6) A frame to display the inverse matrix (if computed)
        self.results_frame = tk.Frame(self)



    def create_matrix_entries(self):
        """
        Reads n from entry_n, clears the old matrix entries, and creates a new n x n grid.
        """
        # Clear old entries
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.destroy()
        self.entries_matrix.clear()

        # Also clear any old results
        self.clear_results_display()

        # Parse the new size
        try:
            n = int(self.entry_n.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a positive integer for matrix order.")
            return
        
        # Create the new grid of entry widgets
        for i in range(n):
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.matrix_frame, width=8)
                e.grid(row=i, column=j, padx=5, pady=5)
                row_entries.append(e)
            self.entries_matrix.append(row_entries)

    def on_calculate(self):
        """
        Gathers the user-input matrix from the entries, then calls
        an inverse function from MatrixOperations, and displays the result.
        """
        n = len(self.entries_matrix)
        if n == 0:
            messagebox.showwarning("No matrix", "Please set matrix size and enter values first.")
            return

        # Build the matrix A from the entries
        A = []
        try:
            for i in range(n):
                row_vals = []
                for j in range(n):
                    val_str = self.entries_matrix[i][j].get()
                    val = float(val_str)
                    row_vals.append(val)
                A.append(row_vals)
        except ValueError:
            messagebox.showerror("Invalid input", "All matrix entries must be numeric.")
            return
        eps = machine_epsilon()
        print("eps: ", eps)
        inv_mat, steps = MatrixOperations.gauss_jordan_inverse(A,eps)
        self.text_steps.delete("1.0", tk.END)
        # מציגים את השלבים
        for step_text in steps:
            self.text_steps.insert(tk.END, step_text + "\n\n")

        if inv_mat is None:
            # לא ניתן למצוא הופכית
            self.text_steps.insert(tk.END, "לא נמצאה מטריצה הופכית (כנראה דטרמיננטה=0)\n")
            return

        # מציגים את ההופכית
        self.text_steps.insert(tk.END, "המטריצה ההופכית (A^-1):\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(inv_mat) + "\n\n")

        # בודקים את הכפל A*A^-1 ו־A^-1*A
        prod1 = MatrixOperations.multiply_matrices(A, inv_mat)
        prod2 = MatrixOperations.multiply_matrices(inv_mat, A)

        self.text_steps.insert(tk.END, "A * A^-1:\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(prod1) + "\n")
        self.text_steps.insert(tk.END, "A^-1 * A:\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(prod2) + "\n\n")

        check1 = MatrixOperations.is_identity(prod1,eps)
        check2 = MatrixOperations.is_identity(prod2,eps)

        if check1 and check2:
            self.text_steps.insert(tk.END, "הכפל A*A^-1 וגם A^-1*A נתנו מטריצת יחידה. ההופכית כנראה נכונה!\n")
        else:
            self.text_steps.insert(tk.END, "נראה שהכפל לא נתן את מטריצת היחידה. ייתכן שהייתה טעות.\n")
        # 7) Display the result in a green frame
        self.display_results(inv_mat)

    def on_clear(self):
        """
        Clears the matrix entries and also the results area, but does not reset n.
        """
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.delete(0, tk.END)
        self.text_steps.delete("1.0", tk.END)
        self.clear_results_display()

    def clear_results_display(self):
        """
        Removes or clears the results frame.
        """
        
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.results_frame.pack_forget()

    def display_results(self, inv_mat):
        """
        Creates a green panel showing the heading "INVERSE MATRIX" and each cell
        of the inverse in a table-like format.
        """
        # Clear anything old
        self.clear_results_display()

        self.results_frame.config(bg="#43A047")  # a green shade
        self.results_frame.pack(fill=tk.X, pady=10)

        label_header = tk.Label(self.results_frame, text="INVERSE MATRIX", 
                                font=("Courier", 14, "bold"), bg="#43A047", fg="white")
        label_header.pack(pady=5)

        # Now show the data in a grid
        # We'll create a sub-frame so that the green background is behind it
        data_frame = tk.Frame(self.results_frame, bg="#43A047")
        data_frame.pack(padx=10, pady=10)

        n = len(inv_mat)
        for i in range(n):
            for j in range(n):
                val = inv_mat[i][j]
                # Format to 3 decimals, for example
                label_val = tk.Label(data_frame, text=f"{val:6.3f}", 
                                     bg="#43A047", fg="white", width=8, anchor="e")
                label_val.grid(row=i, column=j, padx=5, pady=2)

class GaussianEliminationFrame(tk.Frame):
    """
    Frame for entering an augmented matrix (n x (n+1)) for a system of n unknowns,
    then computing and displaying the solution using Gaussian Elimination.
    """
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # 1) A label + entry for "Choose Number Of Unknowns"
        self.label_n = tk.Label(self, text="Choose Number Of Unknowns", font=("Courier", 10, "bold"))
        self.label_n.pack(pady=(5,0))

        self.entry_n = tk.Entry(self, width=5)
        self.entry_n.pack(pady=(0,10))
        # Start empty (or default to "3" for convenience)
        # self.entry_n.insert(0, "")

        # 2) Button to build the augmented matrix
        self.btn_update = tk.Button(self, text="Update Matrix Size", command=self.create_matrix_entries)
        self.btn_update.pack(pady=(0,10))

        # 3) A label for "Enter Coefficients of Augmented Matrix"
        self.label_title = tk.Label(self, text="Enter Coefficients of Augmented Matrix:", 
                                    font=("Courier", 14, "bold"))
        self.label_title.pack()

        # 4) A frame for the augmented matrix’s Entry widgets
        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.pack(pady=5)

        # This 2D list will be of shape n x (n+1) for the user’s input
        self.entries_matrix = []

        # 5) A bottom frame for the Calculate/Clear buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        self.btn_calculate = tk.Button(btn_frame, text="CALCULATE", bg="#4CAF50", fg="white",
                                       command=self.on_calculate)
        self.btn_calculate.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(btn_frame, text="CLEAR", bg="#B71C1C", fg="white",
                                   command=self.on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.text_steps = tk.Text(self, height=20, width=80)
        self.text_steps.pack(pady=10)
        # 6) A frame to show the solution
        self.results_frame = tk.Frame(self)
        # We will pack/place it only once we have a result.

    def create_matrix_entries(self):
        """
        Reads n from self.entry_n, clears old matrix entries,
        and creates a new grid of n x (n+1) entry widgets for the augmented matrix.
        """
        # Clear old entries
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.destroy()
        self.entries_matrix.clear()
        self.clear_results_display()

        # Parse new size n
        try:
            n = int(self.entry_n.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a positive integer for number of unknowns.")
            return

        # Create n x (n+1) Entry widgets
        for i in range(n):
            row_entries = []
            for j in range(n+1):  # n+1 columns for augmented matrix
                e = tk.Entry(self.matrix_frame, width=8)
                e.grid(row=i, column=j, padx=5, pady=5)
                row_entries.append(e)
            self.entries_matrix.append(row_entries)


    def on_calculate(self):
        """
        Reads the augmented matrix from the entries, calls gauss elimination solver,
        and displays the solution vector (x0, x1, etc.).
        """
        n = len(self.entries_matrix)
        if n == 0:
            messagebox.showwarning("No matrix", "Please set the dimension and enter the augmented matrix first.")
            return

        # Build the augmented matrix
        A = []
        b = []
        try:
            for i in range(n):
                row_vals = []
                for j in range(n):
                    val_str = self.entries_matrix[i][j].get()
                    val = float(val_str)
                    row_vals.append(val)
                A.append(row_vals)
                val_str = self.entries_matrix[i][n].get()
                val = float(val_str)
                b.append(val)
        except ValueError:
            messagebox.showerror("Invalid input", "All matrix entries must be numeric.")
            return

        # ננקה את תיבת התצוגה
        self.text_steps.delete("1.0", tk.END)

        # שומרים עותק של A המקורית לבדיקה אח"כ
        A_orig = MatrixOperations.copy_matrix(A)

        # 1) מפעילים את האלגוריתם
        steps, invertible = MatrixOperations.gauss_jordan_with_elementary_matrices(A)

        # 2) מדפיסים שלבים (המטריצות האלמנטריות)
        self.text_steps.insert(tk.END, "===== פעולות אלמנטריות =====\n")
        for idx, (E, desc) in enumerate(steps, start=1):
            self.text_steps.insert(tk.END, f"\n--- שלב {idx}: {desc} ---\n")
            self.text_steps.insert(tk.END, "מטריצה אלמנטרית E:\n")
            self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(E) + "\n")

        # 3) אם A אינה הפיכה
        if not invertible:
            self.text_steps.insert(tk.END, "\nA אינה הפיכה (דטרמיננטה=0).\n")
            return

        # 4) חושבים על A^-1 כמכפלת כל המטריצות האלמנטריות (E_k ... E_1)
        A_inv = MatrixOperations.build_identity(n)
        for (E, _) in steps:
            A_inv = MatrixOperations.multiply_matrices(E, A_inv)

        self.text_steps.insert(tk.END, "\n===== A^-1 (המטריצה ההופכית) =====\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(A_inv) + "\n")

        # 5) בדיקות: A*A^-1 ו-A^-1*A
        prod1 = MatrixOperations.multiply_matrices(A_orig, A_inv)
        prod2 = MatrixOperations.multiply_matrices(A_inv, A_orig)

        self.text_steps.insert(tk.END, "\nבדיקה: A * A^-1:\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(prod1) + "\n")
        self.text_steps.insert(tk.END, f"האם זו I? {MatrixOperations.is_identity(prod1)}\n")

        self.text_steps.insert(tk.END, "\nבדיקה: A^-1 * A:\n")
        self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(prod2) + "\n")
        self.text_steps.insert(tk.END, f"האם זו I? {MatrixOperations.is_identity(prod2)}\n")

        # 6) פתרון Ax = b => x = A^-1 * b
        x = MatrixOperations.multiply_matrix_vector(A_inv, b)

        self.text_steps.insert(tk.END, "\n===== פתרון המערכת Ax = b =====\n")
        self.text_steps.insert(tk.END, "x = A^-1 * b:\n")
        for i, val in enumerate(x):
            self.text_steps.insert(tk.END, f"x[{i}] = {val:.5f}\n")

        # בודקים A*x ~ b
        Ax = MatrixOperations.multiply_matrix_vector(A_orig, x)
        self.text_steps.insert(tk.END, "\nבדיקה: A*x (צריך להיות קרוב ל-b)\n")
        self.text_steps.insert(tk.END, f"A*x = {Ax}\n")
        self.text_steps.insert(tk.END, f"b    = {b}\n")
        close_check = MatrixOperations.is_close_to_vector(Ax, b)
        self.text_steps.insert(tk.END, f"האם קרוב? {close_check}\n")

        if close_check is None:
            messagebox.showerror("No unique solution", "The system might be singular or have infinite solutions.")
            return

        # 7) If we have a solution array, display it
        self.display_solution(x)

    def on_clear(self):
        """
        Clears all matrix entries and any displayed solution, but doesn't reset n.
        """
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.delete(0, tk.END)
        self.text_steps.delete("1.0", tk.END)
        self.clear_results_display()

    def clear_results_display(self):
        """
        Removes/clears the results frame.
        """
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.results_frame.pack_forget()

    def display_solution(self, solution):
        """
        Creates a green panel showing each x_i in a list.
        """
        self.clear_results_display()

        self.results_frame.config(bg="#43A047")  # a green shade
        self.results_frame.pack(fill=tk.X, pady=10)

        label_header = tk.Label(self.results_frame, text="SOLUTION", 
                                font=("Courier", 14, "bold"), bg="#43A047", fg="white")
        label_header.pack(pady=5)

        data_frame = tk.Frame(self.results_frame, bg="#43A047")
        data_frame.pack(padx=10, pady=10)

        # solution is presumably a list of floats [x0, x1, ..., x_{n-1}]
        for i, val in enumerate(solution):
            label_val = tk.Label(data_frame, text=f"x{i} = {val:6.3f}",
                                 bg="#43A047", fg="white", font=("Courier", 12))
            label_val.pack(anchor="w")

class GaussianEliminationLUFrame(tk.Frame):
    """
    Frame for LU decomposition using Gaussian Elimination.
    The user enters an n x n matrix, performs LU factorization, 
    and displays L and U matrices along with detailed steps.
    """
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # 1) Label + entry for matrix order
        self.label_n = tk.Label(self, text="Choose Order Of Matrix", font=("Courier", 10, "bold"))
        self.label_n.pack(pady=(5, 0))

        self.entry_n = tk.Entry(self, width=5)
        self.entry_n.pack(pady=(0, 10))

        # 2) Button to build the entry grid
        self.btn_update = tk.Button(self, text="Update Matrix Size", command=self.create_matrix_entries)
        self.btn_update.pack(pady=(0, 10))

        # 3) Label for entering matrix coefficients
        self.label_title = tk.Label(self, text="Enter Coefficients of Matrix:", font=("Courier", 14, "bold"))
        self.label_title.pack()

        # 4) Frame for matrix entry widgets
        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.pack(pady=5)

        self.entries_matrix = []  # 2D list of Entry widgets

        # 5) Bottom frame for Calculate / Clear buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        self.btn_calculate = tk.Button(btn_frame, text="CALCULATE", bg="#4CAF50", fg="white",
                                       command=self.on_calculate)
        self.btn_calculate.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(btn_frame, text="CLEAR", bg="#B71C1C", fg="white",
                                   command=self.on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        # 6) Text widget to display steps
        self.text_steps = tk.Text(self, height=20, width=80)
        self.text_steps.pack(pady=10)

        # 7) A frame to display L and U
        self.results_frame = tk.Frame(self)

    def create_matrix_entries(self):
        """
        Reads matrix order from entry_n, clears old matrix entries,
        and creates a new n x n grid of entry widgets.
        """
        # Clear old entries
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.destroy()
        self.entries_matrix.clear()
        self.clear_results_display()

        # Parse matrix order
        try:
            n = int(self.entry_n.get())  
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a positive integer for matrix order.")
            return

        # Create n x n entries
        for i in range(n):
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.matrix_frame, width=8)
                e.grid(row=i, column=j, padx=5, pady=5)
                row_entries.append(e)
            self.entries_matrix.append(row_entries)

    def on_calculate(self):
        """
        קורא את המטריצה, מבצע פירוק LU עם Pivoting מלא, ומציג את L, U והשלבים.
        """
        try:
            n = int(self.entry_n.get())  
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("שגיאה", "יש להזין מספר שלם וחיובי לגודל המטריצה.")
            return

        # בניית מטריצת A מהקלט
        A = []
        try:
            for i in range(n):
                row_vals = []
                for j in range(n):
                    val_str = self.entries_matrix[i][j].get()
                    val = float(val_str)
                    row_vals.append(val)
                A.append(row_vals)
        except ValueError:
            messagebox.showerror("שגיאה", "כל הערכים במטריצה חייבים להיות מספרים תקינים.")
            return

        # ניקוי תיבת התצוגה
        self.text_steps.delete("1.0", tk.END)

        # ביצוע פירוק LU עם טיפול בשגיאות
        try:
            L, U, perm_cols, steps = self.compute_lu_with_steps(A)
            valid, diff = MatrixOperations.check_lu_decomposition(A, L, U, perm_cols)
        except ValueError as e:
            messagebox.showerror("שגיאה", str(e))
            return

        # הצגת שלבי הפירוק
        self.text_steps.insert(tk.END, "===== שלבי פירוק LU =====\n")
        for step in steps:
            self.text_steps.insert(tk.END, step + "\n")

        if valid:
            self.text_steps.insert(tk.END, " פירוק LU תקין!\n")
            self.display_lu(L, U)
        else:
            self.text_steps.insert(tk.END, " פירוק LU שגוי!\n")
            self.text_steps.insert(tk.END, "הפרש בין A לבין P^(-1) * L * U:\n")
            self.text_steps.insert(tk.END, MatrixOperations.matrix_to_string(diff) + "\n")



    def on_clear(self):
        """
        Clears the matrix entries and any displayed L/U matrices and steps.
        """
        for row_entries in self.entries_matrix:
            for e in row_entries:
                e.delete(0, tk.END)

        self.text_steps.delete("1.0", tk.END)
        self.clear_results_display()

    def clear_results_display(self):
        """
        Clears the results frame.
        """
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.results_frame.pack_forget()

    def display_lu(self, L, U):
        """
        Creates a panel showing L and U in a table-like format.
        """
        self.clear_results_display()
        self.results_frame.config(bg="#43A047")
        self.results_frame.pack(fill=tk.X, pady=10)

        # Label for L
        label_L = tk.Label(self.results_frame, text="Lower Triangular Matrix (L):",
                           font=("Courier", 14, "bold"), bg="#43A047", fg="white")
        label_L.pack(pady=5)

        # Frame for L
        frame_L = tk.Frame(self.results_frame, bg="#43A047")
        frame_L.pack(padx=10, pady=10)

        for i in range(len(L)):
            for j in range(len(L)):
                val = L[i][j]
                label_val = tk.Label(frame_L, text=f"{val:6.3f}",
                                     bg="#43A047", fg="white", width=8, anchor="e")
                label_val.grid(row=i, column=j, padx=5, pady=2)

        # Label for U
        label_U = tk.Label(self.results_frame, text="Upper Triangular Matrix (U):",
                           font=("Courier", 14, "bold"), bg="#43A047", fg="white")
        label_U.pack(pady=5)

        # Frame for U
        frame_U = tk.Frame(self.results_frame, bg="#43A047")
        frame_U.pack(padx=10, pady=10)

        for i in range(len(U)):
            for j in range(len(U)):
                val = U[i][j]
                label_val = tk.Label(frame_U, text=f"{val:6.3f}",
                                     bg="#43A047", fg="white", width=8, anchor="e")
                label_val.grid(row=i, column=j, padx=5, pady=2)

    def compute_lu_with_steps(self, matrix):
        """
        Performs LU decomposition with full pivoting and records steps.
        """
        steps = []
        L, U, perm_cols = MatrixOperations.find_LU_matrix(matrix)  # הוספת Pivoting מלא
    
        steps.append("Initial Matrix A:")
        steps.append(MatrixOperations.matrix_to_string(matrix))
    
        steps.append("Lower Triangular Matrix (L):")
        steps.append(MatrixOperations.matrix_to_string(L))
    
        steps.append("Upper Triangular Matrix (U):")
        steps.append(MatrixOperations.matrix_to_string(U))
    
        return L, U, perm_cols, steps  # מחזירים את וקטור ההחלפות

class IterativeJacobiFrame(tk.Frame):
    """Frame for Jacobi Method (iterative)."""
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Iterative Methods - Jacobi").pack(pady=10)
        # Add widgets


class IterativeGaussSeidelFrame(tk.Frame):
    """Frame for Gauss-Seidel iterative method."""
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Iterative Methods - Gauss-Seidel").pack(pady=10)
        # Add widgets

class BisectionMethodFrame(tk.Frame):
    """Frame for the Bisection method."""
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="Bisection Method").pack(pady=10)
        # Add widgets

class MainWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(fill='both', expand=True)

        self.master.title("Numerical Methods - Menu on Left, Calculator on Right")

        # Create two main subframes: left (menu) and right (content)
        self.left_menu = tk.Frame(self, width=200)
        self.left_menu.pack(side=tk.LEFT, fill=tk.Y)

        self.right_content = tk.Frame(self,height=1000,width=1000, bg="white")
        self.right_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create all frames for each topic and store them in a dict
        self.frames = {}
        self.frames['MachinePrecision'] = MachinePrecisionFrame(self.right_content)
        self.frames['InverseMatrix']    = InverseMatrixFrame(self.right_content)
        self.frames['GaussianElim']     = GaussianEliminationFrame(self.right_content)
        self.frames['GaussianElimLU']   = GaussianEliminationLUFrame(self.right_content)
        self.frames['IterativeJacobi']  = IterativeJacobiFrame(self.right_content)
        self.frames['IterativeGSeidel'] = IterativeGaussSeidelFrame(self.right_content)
        self.frames['Bisection']        = BisectionMethodFrame(self.right_content)
        # you can add more frames as you like

        # Initially, hide them or just show none
        for frame in self.frames.values():
            frame.place(in_=self.right_content, x=0, y=0, relwidth=1, relheight=1)
            frame.lower()

        # Build the menu on the left side:
        tk.Button(self.left_menu, text="machine precision", 
                  command=lambda: self.show_frame('MachinePrecision')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Inverse matrix and elementary matrices", 
                  command=lambda: self.show_frame('InverseMatrix')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Gaussian Elimination", 
                  command=lambda: self.show_frame('GaussianElim')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="טיב הצגה ונורמת המקסימום", 
                  command=lambda: messagebox.showinfo("TODO", "Add a frame or function here.")
                  ).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Gaussian Elimination LU", 
                  command=lambda: self.show_frame('GaussianElimLU')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Iterative Methods Jacobi Method", 
                  command=lambda: self.show_frame('IterativeJacobi')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Iterative Methods Gauss-Seidel Method", 
                  command=lambda: self.show_frame('IterativeGSeidel')).pack(pady=5, fill=tk.X)

        tk.Button(self.left_menu, text="Method Bisec", 
                  command=lambda: self.show_frame('Bisection')).pack(pady=5, fill=tk.X)

        # add more as needed

    def show_frame(self, frame_name):
        """
        Raise the selected frame in the right content area.
        """
        frame = self.frames[frame_name]
        frame.lift()  # bring it to the top

def main():
    root = tk.Tk()
    app = MainWindow(root)
    app.mainloop()

if __name__ == "__main__":
    main()
