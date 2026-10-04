package com.example.sudoku;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Base64;

@RestController
@RequestMapping("/api")
public class SudokuController {

    /** puzzle = digits (read from the image, or as corrected by the user); confidence = 0..1 per digit; solution = full solved board; image = solved picture (data URI). */
    public record SolveResponse(boolean solved, String message, int[][] puzzle, double[][] confidence, int[][] solution, String image) {}

    private final SudokuSolver solver;
    private final SudokuImageService images;
    private final ObjectMapper json = new ObjectMapper();

    public SudokuController(SudokuSolver solver, SudokuImageService images) {
        this.solver = solver;
        this.images = images;
    }

    /**
     * file  = the Sudoku picture.
     * board = optional JSON 9x9 array of corrected digits (0 = empty); when present it replaces the digits read from the picture.
     */
    @PostMapping("/solve-image")
    public ResponseEntity<SolveResponse> solveImage(@RequestParam("file") MultipartFile file,
                                                    @RequestParam(value = "board", required = false) String boardJson) {
        try {
            SudokuImageService.Scan scan = images.scan(file.getBytes());
            int[][] puzzle = scan.board();
            double[][] confidence = scan.confidence();

            if (boardJson != null && !boardJson.isBlank()) {
                puzzle = json.readValue(boardJson, int[][].class);
                if (!wellFormed(puzzle)) return fail(400, "The corrected board must be 9x9 with digits 0-9.", null, null);
                confidence = new double[9][9];
                for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) confidence[r][c] = puzzle[r][c] == 0 ? 0 : 1;
            }

            int digits = 0;
            for (int[] row : puzzle) for (int v : row) if (v != 0) digits++;
            if (digits < 17)   // a valid Sudoku has at least 17 clues; fewer means the grid was not read properly
                return fail(422, "Only " + digits + " digits found. Make sure the whole grid, including all four borders, is visible.", puzzle, confidence);
            if (solver.hasConflict(puzzle))
                return fail(422, "Some digits look wrong (duplicates in a row, column or box). Fix the highlighted digits below and press Solve again.", puzzle, confidence);

            int[][] solved = new int[9][];
            for (int i = 0; i < 9; i++) solved[i] = puzzle[i].clone();
            if (!solver.solve(solved))
                return fail(422, "No solution exists for these digits. A digit may be misread - check the grid below.", puzzle, confidence);

            String png = Base64.getEncoder().encodeToString(images.render(scan, puzzle, solved));
            return ResponseEntity.ok(new SolveResponse(true, "Solved!", puzzle, confidence, solved, "data:image/png;base64," + png));
        } catch (IllegalArgumentException e) {
            return fail(400, e.getMessage(), null, null);
        } catch (IOException e) {
            return fail(500, "Could not read the uploaded data.", null, null);
        }
    }

    private boolean wellFormed(int[][] b) {
        if (b == null || b.length != 9) return false;
        for (int[] r : b) {
            if (r == null || r.length != 9) return false;
            for (int v : r) if (v < 0 || v > 9) return false;
        }
        return true;
    }

    private ResponseEntity<SolveResponse> fail(int status, String msg, int[][] puzzle, double[][] confidence) {
        return ResponseEntity.status(status).body(new SolveResponse(false, msg, puzzle, confidence, null, null));
    }
}
