package com.example.sudoku;

import org.springframework.stereotype.Service;

/** Classic LeetCode 37 "Sudoku Solver" backtracking solution. */
@Service
public class SudokuSolver {

    /** True if the digits already break a row, column or box rule. */
    public boolean hasConflict(int[][] b) {
        boolean[][] row = new boolean[9][10], col = new boolean[9][10], box = new boolean[9][10];
        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                int v = b[r][c], k = r / 3 * 3 + c / 3;
                if (v == 0) continue;
                if (row[r][v] || col[c][v] || box[k][v]) return true;
                row[r][v] = col[c][v] = box[k][v] = true;
            }
        }
        return false;
    }

    /** How many digits share a row, column or box with an equal digit (0 = no conflicts). */
    public int conflictDigits(int[][] b) {
        int n = 0;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++) {
                int v = b[r][c];
                if (v == 0) continue;
                boolean dup = false;
                for (int k = 0; k < 9; k++) {
                    if (k != c && b[r][k] == v) dup = true;
                    if (k != r && b[k][c] == v) dup = true;
                }
                int br = r / 3 * 3, bc = c / 3 * 3;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        if ((br + i != r || bc + j != c) && b[br + i][bc + j] == v) dup = true;
                if (dup) n++;
            }
        return n;
    }

    /** Solves in place. Returns false if the puzzle has no solution. */
    public boolean solve(int[][] b) {
        boolean[][] row = new boolean[9][10], col = new boolean[9][10], box = new boolean[9][10];
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                if (b[r][c] != 0) row[r][b[r][c]] = col[c][b[r][c]] = box[r / 3 * 3 + c / 3][b[r][c]] = true;
        return backtrack(b, 0, row, col, box);
    }

    private boolean backtrack(int[][] b, int pos, boolean[][] row, boolean[][] col, boolean[][] box) {
        if (pos == 81) return true;
        int r = pos / 9, c = pos % 9, k = r / 3 * 3 + c / 3;
        if (b[r][c] != 0) return backtrack(b, pos + 1, row, col, box);

        for (int v = 1; v <= 9; v++) {
            if (row[r][v] || col[c][v] || box[k][v]) continue;
            b[r][c] = v;
            row[r][v] = col[c][v] = box[k][v] = true;
            if (backtrack(b, pos + 1, row, col, box)) return true;
            b[r][c] = 0;
            row[r][v] = col[c][v] = box[k][v] = false;
        }
        return false;
    }
}
