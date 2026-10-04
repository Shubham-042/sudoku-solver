# Sudoku Solver

🚀 **Live Demo:** https://sudoku-solver-cy5h.onrender.com
# Sudoku Solver from Image (Spring Boot)

Upload a photo or screenshot of an unsolved Sudoku (printed OR handwritten) -> get the same picture back with the solution written in.

Requirements: JDK 17+ and Maven (IntelliJ bundles Maven). Internet is needed the first time so Maven can download libraries.

    mvn clean spring-boot:run          # or run SudokuApplication in IntelliJ
    open http://localhost:8080

## How it works
1. GridFinder   - OpenCV finds the grid in the picture (several strategies, each checked by counting grid lines):
                  screenshots, tilted photos, dark mode, background artwork, thin/faint lines, drawn grids.
2. Orientation  - the grid is read in up to 4 rotations; the most plausible reading wins (sideways / upside-down photos work).
3. DigitReader  - grid lines are removed, each cell is cut out and sent to DigitClassifier.
4. DigitClassifier - small neural-network ensemble in pure Java (no Tesseract, no native OCR),
                  trained on MNIST handwriting + ~340 fonts (see training/).
5. SudokuSolver - LeetCode 37 backtracking.
6. SudokuImageService.render - draws the solved digits (green) onto the original picture, same size as the upload.

## Web app (installable)
Sudoku Vision UI: drag & drop / choose / paste / camera, before-after slider, animated solution board,
editable "Check digits" grid (unsure digits highlighted), download + share, light/dark theme.
It is a PWA: in Chrome/Edge use the "Install app" button (or the install icon in the address bar) to get a standalone app window.

### Use it from your phone
1. PC and phone on the same Wi-Fi. Find the PC's IP (Windows: `ipconfig`), open `http://<PC-IP>:8080` on the phone.
   (Allow Java through Windows Firewall if asked.) The "Take photo" button opens the phone camera.
2. "Install app" on a phone needs HTTPS. Easiest: `cloudflared tunnel --url http://localhost:8080` or ngrok gives a free https link,
   or deploy the jar to a server.

## API
POST /api/solve-image   multipart: file (image), board (optional JSON 9x9 of corrected digits, 0 = empty)
-> {"solved":true,"message":"Solved!","puzzle":[[..]],"confidence":[[..]],"solution":[[..]],"image":"data:image/png;base64,..."}

## Tips
Whole grid with all four borders in view, reasonably sharp, not too dark. Very messy handwriting may need a
correction in the editable grid.
