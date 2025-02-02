import numpy as np

class MatrixOperations:

    # =========================================================================
    # 1. פונקציות עזר כלליות
    # =========================================================================

    @staticmethod
    def build_identity(n):
        """
        בונה מטריצת יחידה n x n
        """
        I = [[0]*n for _ in range(n)]
        for i in range(n):
            I[i][i] = 1
        return I

    @staticmethod
    def copy_matrix(M):
        """
        מחזיר עותק (Copy) של מטריצה דו־ממדית
        """
        return [row[:] for row in M]
    @staticmethod
    def vector_1_norm(vector):
        return np.sum(np.abs(vector))

    @staticmethod
    def vector_inf_norm(vector):
        return np.max(np.abs(vector))

    @staticmethod
    def one_matrix_norm(matrix):
        return np.max(np.sum(np.abs(matrix), axis=0))

    @staticmethod
    def infinity_matrix_norm(matrix):
        return np.max(np.sum(np.abs(matrix), axis=1))

    @staticmethod
    def condA(matrix):
        matrix_inv = np.linalg.inv(matrix)
        norm_A = MatrixOperations.infinity_matrix_norm(matrix)
        norm_A_inv = MatrixOperations.infinity_matrix_norm(matrix_inv)
        return norm_A * norm_A_inv

    @staticmethod
    def find_LU_matrix(matrix, eps=1e-7):
        
        n = len(matrix)
        umatrix = np.array(MatrixOperations.copy_matrix(matrix), dtype=float)  
        lmatrix = np.eye(n, dtype=float)  
        perm_cols = np.arange(n)  

        for i in range(n):
            # **Step 3: Gaussian elimination - אפס מתחת לאלכסון**
            for j in range(i + 1, n):
                factor = umatrix[j, i] / umatrix[i, i]
                umatrix[j] -= factor * umatrix[i]
                lmatrix[j, i] = factor  

        return lmatrix, umatrix, perm_cols  # מחזירים גם את וקטור ההחלפות
    

    @staticmethod
    def check_lu_decomposition(A, L, U, perm_cols, eps=1e-7):
        """
        בודק אם פירוק LU תקין על ידי חישוב P^(-1) * L * U והשוואתו ל-A.
        """
        A = np.array(A, dtype=float)
        L = np.array(L, dtype=float)
        U = np.array(U, dtype=float)
    
        # יצירת מטריצת P^-1 מהחלפות עמודות
        P_inv = np.eye(len(A))[:, perm_cols]
    
        # חישוב P^(-1) * L * U
        LU_product = P_inv @ L @ U
    
        # חישוב ההפרש בין A ל-(P^-1 * L * U)
        error = np.abs(A - LU_product)
    
        # בדיקה האם כל האיברים בהפרש קטנים מהסף
        valid = np.all(error < eps)
        return valid, error  # מחזיר האם הפירוק תקין ואת השגיאה


    @staticmethod
    def solve_system_with_LU(A, b):
        lmatrix, umatrix = MatrixOperations.find_LU_matrix(A)
        y = np.linalg.solve(lmatrix, b)
        x = np.linalg.solve(umatrix, y)
        return x
    @staticmethod
    def matrix_to_string(M, aug_n=None):
        """
        המרת מטריצה למחרוזת נעימה לקריאה.
        אם aug_n != None והקיים M היא מטריצה מאוגמנטת [A|B],
        ניתן להציג קו מפריד בעמודה aug_n (לא חובה לשימוש).
        """
        if aug_n is not None and len(M[0]) >= 2*aug_n:
            # הדפסה בסגנון [A|B]
            rows_str = []
            for row in M:
                left_part = row[:aug_n]
                right_part = row[aug_n:]
                rows_str.append(
                    "[" + " ".join(f"{val:7.3f}" for val in left_part)
                    + " | "
                    + " ".join(f"{val:7.3f}" for val in right_part)
                    + "]"
                )
            return "\n".join(rows_str)
        else:
            # מטריצה רגילה
            rows_str = []
            for row in M:
                rows_str.append("[" + " ".join(f"{val:8.3f}" for val in row) + "]")
            return "\n".join(rows_str)

    # =========================================================================
    # 2. פעולות על מטריצות ווקטורים
    # =========================================================================

    @staticmethod
    def multiply_matrices(A, B):
        """
        כפל שתי מטריצות ריבועיות בגודל n x n ומחזיר מטריצת תוצאה.
        מניח ש־A ו־B ריבועיות באותו גודל n x n.
        """
        n = len(A)
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(n):
                    s += A[i][k] * B[k][j]
                C[i][j] = s
        return C

    @staticmethod
    def multiply_matrix_vector(A, v):
        """
        כפל מטריצה (n x n) בווקטור (אורך n) => מחזיר וקטור אורך n.
        """
        n = len(A)
        result = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                s += A[i][j] * v[j]
            result[i] = s
        return result

    @staticmethod
    def is_identity(mat, eps=1e-7):
        """
        בדיקה האם מטריצה ריבועית היא מטריצת היחידה, בהינתן מרווח טעות (eps).
        """
        n = len(mat)
        for i in range(n):
            for j in range(n):
                expected = 1 if i == j else 0
                if abs(mat[i][j] - expected) > eps:
                    return False
        return True

    @staticmethod
    def is_close_to_vector(u, v, eps=1e-7):
        """
        בדיקה האם שני וקטורים (אורך n) קרובים זה לזה עד eps.
        """
        if len(u) != len(v):
            return False
        for i in range(len(u)):
            if abs(u[i] - v[i]) > eps:
                return False
        return True

    # =========================================================================
    # 3. Gauss-Jordan להופכית (בצורה ישירה - יוצרת מטריצה מאוגמנטת)
    # =========================================================================

    @staticmethod
    def gauss_jordan_inverse(A,eps=1e-14):
        """
        מקבל מטריצה ריבועית A (רשימות רשימות), מחזיר:
        1) מטריצה A^-1 (אם קיימת) 
        2) steps: רשימת מחרוזות שתתאר את השלבים

        במידה והדטרמיננטה 0 או יש בעיה בדרך - יוחזר (None, steps).
        """
        ops = []
        n = len(A)
        # בניית מטריצה מאוגמנטת [A|I]
        M = []
        for i in range(n):
            row = A[i][:]
            identity_part = [0]*n
            identity_part[i] = 1
            row += identity_part
            M.append(row)

        ops.append("התחלת מטריצה מאוגמנטת [A | I]:\n" + MatrixOperations.matrix_to_string(M, n))

        # אלגוריתם
        for i in range(n):
            # pivot
            pivot = M[i][i]
            if abs(pivot) < eps:
                # לחפש שורה עם פיבוט גדול
                for r in range(i+1, n):
                    if abs(M[r][i]) > abs(pivot):
                        M[i], M[r] = M[r], M[i]
                        ops.append(f"החלפנו שורה {i} בשורה {r} כי הפיבוט היה קטן מדי:\n" + MatrixOperations.matrix_to_string(M, n))
                        pivot = M[i][i]
                        break
            if abs(pivot) < eps:
                ops.append("סינגולרית (pivot=0). אין הופכית.")
                return None, ops
            
            # scale pivot to 1
            inv_pivot = 1.0/pivot
            for c in range(2*n):
                M[i][c] *= inv_pivot
            ops.append(f"נרמלנו את השורה {i}:\n" + MatrixOperations.matrix_to_string(M, n))

            # אפס בכל עמודה i חוץ משורה i
            for r in range(n):
                if r != i:
                    factor = M[r][i]
                    for c in range(2*n):
                        M[r][c] -= factor * M[i][c]
                    ops.append(f"ניקינו את העמודה {i} בשורה {r}:\n" +
                               MatrixOperations.matrix_to_string(M, n))

        # שליפת החלק הימני
        inv_mat = []
        for i in range(n):
            inv_mat.append(M[i][n:])

        return inv_mat, ops

    # =========================================================================
    # 4. Gauss-Jordan ע"י פעולות אלמנטריות (שומר רשימת מטריצות אלמנטריות)
    # =========================================================================

    @staticmethod
    def gauss_jordan_with_elementary_matrices(A,eps=1e-14):
        """
        מקבלת מטריצה A (רשימות רשימות) בגודל n x n.
        מפעילה אלגוריתם Gauss-Jordan מלא, תוך שימוש ב"מטריצות אלמנטריות" בכל צעד.
        
        מחזירה:
        1) רשימת (E_i, description)
        2) bool האם הצלחנו להגיע למטריצת היחידה (True=הפיכה, False=סינגולרית)

        להשלמת החישוב של A^-1 יש לכפול את כל E_i בסדר שקיבלנו.
        """
        n = len(A)
        M = MatrixOperations.copy_matrix(A)  # עותק
        steps = []

        for i in range(n):
            pivot = M[i][i]
            if abs(pivot) < eps:
                # לחפש שורה אחרת
                max_row = i
                max_val = abs(pivot)
                for r in range(i+1, n):
                    if abs(M[r][i]) > max_val:
                        max_val = abs(M[r][i])
                        max_row = r
                if max_row != i:
                    E_swap = MatrixOperations.elementary_swap(n, i, max_row)
                    M = MatrixOperations.multiply_matrices(E_swap, M)
                    steps.append((E_swap, f"החלפת שורות {i} ו-{max_row}"))
                    pivot = M[i][i]

            if abs(pivot) < eps:
                return steps, False

            # נרמל את השורה i (pivot=1)
            scale_factor = 1.0/M[i][i]
            E_scale = MatrixOperations.elementary_scale(n, i, scale_factor)
            M = MatrixOperations.multiply_matrices(E_scale, M)
            steps.append((E_scale, f"כפל שורה {i} ב-{scale_factor:.5f}"))

            # אפס ביתר העמודה
            for r in range(n):
                if r != i:
                    factor = -M[r][i]
                    if abs(factor) > eps:
                        E_add = MatrixOperations.elementary_add_row(n, i, r, factor)
                        M = MatrixOperations.multiply_matrices(E_add, M)
                        steps.append((E_add, f"שורה {r} = שורה {r} + ({factor:.5f})*שורה {i}"))

        return steps, True

    # =========================================================================
    # 5. פעולות אלמנטריות (להשלמת Gauss-Jordan וכו')
    # =========================================================================

    @staticmethod
    def elementary_swap(n, r1, r2):
        """
        מחזירה מטריצה אלמנטרית E המבצעת החלפת שורות r1 ו-r2
        """
        E = MatrixOperations.build_identity(n)
        E[r1], E[r2] = E[r2], E[r1]
        return E

    @staticmethod
    def elementary_scale(n, r, factor):
        """
        מטריצה אלמנטרית: כפל שורה r בסקלר factor
        """
        E = MatrixOperations.build_identity(n)
        E[r][r] = factor
        return E

    @staticmethod
    def elementary_add_row(n, src, dest, factor):
        """
        מטריצה אלמנטרית: dest <- dest + factor*src
        """
        E = MatrixOperations.build_identity(n)
        E[dest][src] = factor
        return E

    # =========================================================================
    # 6. פונקציות נוספות (Determinant, LU decomposition, וכו') - אופציונלי
    # =========================================================================

    @staticmethod
    def determinant(A):
        """
        דוגמה לפונקציה לחישוב דטרמיננטה (לפי אלגוריתם שתבחר).
        כאן רק הדגמה קצרה (אפשר לממש כרצונך).
        """
        # אפשר לממש ע"י Gauss Elimination למשל, להחזיר -1 אם סינגולרית
        pass

    @staticmethod
    def partial_pivoting(A, i):
        """
        דוגמה לפונקציה (אופציונלית) שמבצעת partial pivoting על השורה i במטריצה A.
        """
        pass

    # ... וכו'. אפשר להוסיף לפי הצורך.
