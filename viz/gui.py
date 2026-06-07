# Copyright (C) 2026  Nate MacFadden
#
# tkinter GUI for stepping through Befunge programs

import os
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont

import numpy as np

import befunge as bf
from viz import graph


class Interpreter:
    """
    Drives befunge._run_core one step at a time; carries no dispatch logic
    of its own.
    """

    def __init__(self, src=""):
        """
        Build an interpreter and load `src` (empty grid by default)
        """
        self.load(src)

    def load(self, src):
        """
        Reset all interpreter state and load `src` into a fresh grid
        """
        self.grid    = bf.str_to_grid(src)
        self.regs    = {}  # unbounded register tape: written index -> value
        self._stack   = np.zeros(bf.STACK_CAP,  dtype=np.int64)
        self._out_buf = np.zeros(bf.OUTPUT_CAP, dtype=np.int32)
        # the core requires a visited buffer; the GUI doesn't use the trace
        self._visited = np.zeros(self.grid.shape, dtype=np.uint8)
        self.state    = bf.new_state()
        self.halted   = False
        self.error    = None

    def snapshot(self):
        """
        Capture a deep copy of the full interpreter state for undo
        """
        sp = int(self.state[bf.S_SP])
        n = int(self.state[bf.S_OUT_LEN])
        return (self.grid.copy(),
                self.regs.copy(),
                self._stack[:sp].copy(),
                self._out_buf[:n].copy(),
                self.state.copy(),
                self.halted,
                self.error)

    def restore(self, snap):
        """
        Write a `snapshot` tuple back into the live interpreter state
        """
        g, rg, s, ob, st, h, e = snap
        self.grid[:] = g
        self.regs = dict(rg)
        self.state[:] = st
        self._stack[:len(s)] = s
        self._out_buf[:len(ob)] = ob
        self.halted = h
        self.error = e

    def step(self):
        """
        Advances the interpreter by a single instruction.

        Delegates one step to `befunge._run_core`. Any exception from the core,
        or a non-positive status, halts the interpreter and records the reason
        in `error`.

        Returns
        -------
        None
            All effects are applied in place to the interpreter state.
        """
        if self.halted:
            return
        try:
            status = bf._run_core(
                self.grid, 1, self._stack, self._out_buf, self.state,
                self._visited, self.regs,
            )
        except Exception as e:
            self.halted = True
            self.error = str(e)
            return
        if status == 0:
            self.halted = True
        elif status == 2:
            self.halted = True
            self.error = "p: stack underflow"

    @property
    def stack(self):
        """
        The current stack as a python list, bottom to top.
        """
        return [int(v) for v in self._stack[:int(self.state[bf.S_SP])]]

    @property
    def output(self):
        """
        The accumulated program output, decoded as a string.
        """
        n = int(self.state[bf.S_OUT_LEN])
        return ''.join(chr(int(c)) for c in self._out_buf[:n])

    @property
    def x(self):
        return int(self.state[bf.S_X])

    @property
    def y(self):
        return int(self.state[bf.S_Y])

    @property
    def dx(self):
        return int(self.state[bf.S_DX])

    @property
    def dy(self):
        return int(self.state[bf.S_DY])

    @property
    def string_mode(self):
        return bool(self.state[bf.S_STRING_MODE])


class BefungeGrid(tk.Frame):
    """
    A WxH grid of cells on a Canvas, optionally editable. Each cell is drawn
    independently so content changes never reflow the layout.
    """

    GRID_LINE      = 'gray80'
    BG             = 'white'
    IP_COLOR       = 'gold'
    CURSOR_OUTLINE = 'dodgerblue'

    def __init__(self, parent, cols=bf.W, rows=bf.H, cell_w=9, cell_h=14,
                 font=None, editable=False, on_change=None):
        """
        Builds a fixed WxH grid of cells on a Canvas.

        Draws the static 1px grid lines once, allocates the per-cell logical
        content buffer, and (when editable) wires up click/key/paste bindings
        and an initial cursor. Cells are drawn as independent canvas items so a
        content change never reflows the layout.

        Parameters
        ----------
        parent : tk.Widget
            The parent widget this frame is packed into.
        Cols, rows : int
            Grid dimensions in cells; default to the interpreter's W, H.
        Cell_w, cell_h : int
            Per-cell pixel width and height.
        Font : tkfont.Font, optional
            Font for cell glyphs; defaults to TkFixedFont.
        Editable : bool
            When True, the grid accepts clicks, keystrokes, and pastes.
        On_change : callable, optional
            Called with no args whenever an edit mutates a cell.

        Returns
        -------
        None
        """
        super().__init__(parent, bd=0, padx=0, pady=0)
        self.cols      = cols
        self.rows      = rows
        self.cell_w    = cell_w
        self.cell_h    = cell_h
        self.editable  = editable
        self.on_change = on_change
        self.font      = font or tkfont.nametofont("TkFixedFont")

        cw = cols * cell_w + 1
        ch = rows * cell_h + 1
        self.canvas = tk.Canvas(self, width=cw, height=ch, bg=self.BG,
                                bd=0, highlightthickness=0, takefocus=editable)
        self.canvas.pack()

        # offset by 0.5 for crisp 1px grid lines
        for c in range(cols + 1):
            self.canvas.create_line(c * cell_w + 0.5, 0,
                                    c * cell_w + 0.5, rows * cell_h,
                                    fill=self.GRID_LINE)
        for r in range(rows + 1):
            self.canvas.create_line(0, r * cell_h + 0.5,
                                    cols * cell_w, r * cell_h + 0.5,
                                    fill=self.GRID_LINE)

        # per-cell state: _chars is the logical content, _text_ids the canvas
        # item ids for cells currently drawn
        self._chars         = [[' '] * cols for _ in range(rows)]
        self._text_ids      = {}
        self._ip_rect       = None
        self._cursor_rect   = None
        self._cursor        = (0, 0)

        if editable:
            self.canvas.bind('<Button-1>', self._on_click)
            self.canvas.bind('<Key>', self._on_key)
            # explicit Cmd/Ctrl-v as a fallback in case <<Paste>> doesn't fire
            self.canvas.bind('<<Paste>>',   self._on_paste)
            self.canvas.bind('<Command-v>', self._on_paste)
            self.canvas.bind('<Control-v>', self._on_paste)
            self._draw_cursor()

    # public API
    # ----------
    def load_src(self, src):
        """
        Load a multi-line source string into the grid.
        """
        for y in range(self.rows):
            row = self._chars[y]
            for x in range(self.cols):
                row[x] = ' '
        for y, line in enumerate(src.splitlines()[:self.rows]):
            for x, ch in enumerate(line[:self.cols]):
                self._chars[y][x] = ch
        self._redraw_all()

    def dump_src(self):
        """
        Serialize the grid contents back to a multi-line source string
        """
        return '\n'.join(''.join(row) for row in self._chars)

    def char_grid(self):
        """
        A copy of the grid contents as an (H, W) list of single chars.
        """
        return [row[:] for row in self._chars]

    def update_from_array(self, arr):
        """
        Redraw only the cells that changed (out-of-range values show a
        placeholder).
        """
        for y in range(self.rows):
            row = self._chars[y]
            arr_row = arr[y]
            for x in range(self.cols):
                v = int(arr_row[x])
                ch = chr(v) if 0 <= v < 0x110000 else '?'
                if row[x] != ch:
                    row[x] = ch
                    self._draw_cell(x, y)

    def highlight_ip(self, x, y):
        """
        Move the yellow IP highlight rectangle to cell (x, y)
        """
        cx = x * self.cell_w
        cy = y * self.cell_h
        coords = (cx + 1, cy + 1, cx + self.cell_w, cy + self.cell_h)
        if self._ip_rect is None:
            self._ip_rect = self.canvas.create_rectangle(
                *coords, fill=self.IP_COLOR, outline='')
            self.canvas.tag_lower(self._ip_rect)
        else:
            self.canvas.coords(self._ip_rect, *coords)

    # internal drawing
    # ----------------
    NONPRINT_COLOR = 'gray50'    # bytes with a standard control-picture glyph
    NOGLYPH_COLOR  = 'firebrick'  # bytes with no standard glyph - call them out

    def _draw_cell(self, x, y):
        """
        (re)draws the glyph for a single cell.

        Deletes any existing canvas text for the cell, then picks a glyph and
        color from the cell's byte value: spaces draw nothing, printable
        ASCII/extended ASCII draw in black, control bytes use gray Unicode
        control pictures, and bytes with no standard glyph are flagged red.

        Parameters
        ----------
        x, y : int
            Cell column and row to redraw.

        Returns
        -------
        None
        """
        key = (x, y)
        if key in self._text_ids:
            self.canvas.delete(self._text_ids.pop(key))
        ch = self._chars[y][x]
        o = ord(ch)

        # glyph + color: space invisible; 0-31/127 use Unicode control
        # pictures (gray); C1 area 128-159 has no standard glyph so flag red
        if o == 32:
            return                         # plain space, nothing to draw
        if 33 <= o < 127 or 160 <= o < 256:
            glyph, color = ch, 'black'     # printable ASCII / extended ASCII
        elif 0 <= o < 32:
            glyph, color = chr(0x2400 + o), self.NONPRINT_COLOR
        elif o == 127:
            glyph, color = '␡', self.NONPRINT_COLOR
        elif 128 <= o < 160:
            glyph, color = '·', self.NOGLYPH_COLOR
        else:
            glyph, color = '?', self.NOGLYPH_COLOR

        cx = x * self.cell_w + self.cell_w / 2
        cy = y * self.cell_h + self.cell_h / 2
        self._text_ids[key] = self.canvas.create_text(
            cx, cy, text=glyph, font=self.font, fill=color)

    def _redraw_all(self):
        """
        Clear and redraw every cell glyph from scratch
        """
        for tid in self._text_ids.values():
            self.canvas.delete(tid)
        self._text_ids.clear()
        for y in range(self.rows):
            for x in range(self.cols):
                self._draw_cell(x, y)

    def _draw_cursor(self):
        """
        Move the editing cursor outline to the current cursor cell
        """
        x, y = self._cursor
        cx = x * self.cell_w
        cy = y * self.cell_h
        coords = (cx + 1, cy + 1, cx + self.cell_w, cy + self.cell_h)
        if self._cursor_rect is None:
            self._cursor_rect = self.canvas.create_rectangle(
                *coords, outline=self.CURSOR_OUTLINE, width=2)
        else:
            self.canvas.coords(self._cursor_rect, *coords)

    # editing
    # -------
    def _on_click(self, event):
        """
        Move the cursor to the clicked cell and grab focus
        """
        x = int(event.x) // self.cell_w
        y = int(event.y) // self.cell_h
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self._cursor = (x, y)
            self._draw_cursor()
            self.canvas.focus_set()

    def _on_key(self, event):
        """
        Handles a keystroke while the grid has focus.

        Arrow keys and Return move the cursor, Backspace/Delete blank a cell,
        and a printable ASCII character is written into the current cell with
        the cursor advancing (wrapping to the next row at the right edge). Any
        edit fires `on_change` and redraws the affected cell.

        Parameters
        ----------
        event : tk.Event
            The key event; `keysym` and `char` drive the behavior.

        Returns
        -------
        None
        """
        x, y = self._cursor
        ks = event.keysym
        if ks == 'Left':
            x = max(0, x - 1)
        elif ks == 'Right':
            x = min(self.cols - 1, x + 1)
        elif ks == 'Up':
            y = max(0, y - 1)
        elif ks == 'Down':
            y = min(self.rows - 1, y + 1)
        elif ks == 'BackSpace':
            x = max(0, x - 1)
            self._chars[y][x] = ' '
            self._draw_cell(x, y)
            if self.on_change:
                self.on_change()
        elif ks == 'Delete':
            self._chars[y][x] = ' '
            self._draw_cell(x, y)
            if self.on_change:
                self.on_change()
        elif ks == 'Return':
            x = 0
            y = min(self.rows - 1, y + 1)
        elif (event.char and len(event.char) == 1
              and 32 <= ord(event.char) < 127):
            self._chars[y][x] = event.char
            self._draw_cell(x, y)
            if self.on_change:
                self.on_change()
            x += 1
            if x >= self.cols:
                x = 0
                y = min(self.rows - 1, y + 1)
        self._cursor = (x, y)
        self._draw_cursor()

    def _on_paste(self, event=None):
        """
        Pastes clipboard text into the grid starting at the cursor.

        Reads the clipboard, then lays characters out from the cursor cell:
        newlines return to the start column on the next row, carriage returns
        are dropped, and lines that overflow the right edge are truncated
        rather than wrapped. Printable ASCII is written; the cursor lands after
        the last written cell and `on_change` fires if anything changed.

        Parameters
        ----------
        event : tk.Event, optional
            The paste event; unused beyond triggering the handler.

        Returns
        -------
        str
            The string ``'break'`` to stop further event propagation.
        """
        try:
            text = self.canvas.clipboard_get()
        except tk.TclError:
            return 'break'
        start_x, start_y = self._cursor
        x, y = start_x, start_y
        changed = False
        for ch in text:
            if ch == '\r':
                continue
            if ch == '\n':
                y += 1
                x = start_x
                continue
            if y >= self.rows:
                break
            if x >= self.cols:
                continue  # don't auto-wrap; let the line truncate
            if 32 <= ord(ch) < 127:
                self._chars[y][x] = ch
                self._draw_cell(x, y)
                changed = True
                x += 1
        if changed:
            self._cursor = (min(x, self.cols - 1), min(y, self.rows - 1))
            self._draw_cursor()
            if self.on_change:
                self.on_change()
        return 'break'


class App:
    def __init__(self, src=""):
        """
        `src` optionally preloads a Befunge source string into the editor -
        handy for `App(df.iloc[i]['pruned_program']).run()` from a notebook.
        """
        self.interp         = Interpreter()
        self.history        = []
        self.running        = False
        self.delay          = 200
        self.steps_per_tick = 1

        self.root = tk.Tk()
        self.root.title("Befunge")

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=9)

        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        # fixed-height headers so both grids start at the same Y; tall enough
        # for a tk.Button on macOS
        HEADER_H = 32

        # LEFT: editor header
        top = tk.Frame(left, height=HEADER_H)
        top.pack(fill="x", anchor="w")
        top.pack_propagate(False)
        tk.Label(top, text="Editor", font=("Sans", 11, "bold")).pack(
            side="left")
        tk.Button(top, text="Load...", command=self.load_file).pack(
            side="right")
        tk.Button(top, text="Clear",   command=self.clear).pack(
            side="right", padx=(0, 4))
        tk.Button(top, text="Show graph", command=self.show_graph).pack(
            side="right", padx=(0, 4))
        self.editor_grid = BefungeGrid(
            left, bf.W, bf.H, font=mono, editable=True, on_change=self.reset)
        self.editor_grid.pack(anchor="w")

        # RIGHT: execution header (status shares the row with the label, to
        # match the editor's label+buttons row)
        exec_top = tk.Frame(right, height=HEADER_H)
        exec_top.pack(fill="x", anchor="w")
        exec_top.pack_propagate(False)
        tk.Label(exec_top, text="Execution", font=("Sans", 11, "bold")).pack(
            side="left")
        self.status = tk.Label(exec_top, text="", font=("Sans", 10), anchor="w")
        self.status.pack(side="left", padx=(8, 0))

        self.display_grid = BefungeGrid(right, bf.W, bf.H, font=mono)
        self.display_grid.pack(anchor="w")

        tk.Label(right, text="Stack (bottom -> top)", font=("Sans", 10, "bold"),
                 anchor="w").pack(fill="x")
        self.stack_view = tk.Text(right, width=bf.W, height=3, font=mono,
                                  wrap="word", state="disabled")
        self.stack_view.pack(fill="x")

        tk.Label(right, text="Data registers (idx: value)",
                 font=("Sans", 10, "bold"), anchor="w").pack(
                     fill="x", pady=(4, 0))
        self.reg_view = tk.Text(right, width=bf.W, height=3, font=mono,
                                wrap="word", state="disabled")
        self.reg_view.pack(fill="x")

        tk.Label(right, text="Output", font=("Sans", 10, "bold"),
                 anchor="w").pack(fill="x", pady=(4, 0))
        out_frame = tk.Frame(right)
        out_frame.pack(fill="x")
        self.output_view = tk.Text(out_frame, width=bf.W, height=8, font=mono,
                                   wrap="char", state="disabled")
        self.output_view.pack(side="left", fill="both", expand=True)
        out_scroll = tk.Scrollbar(out_frame, command=self.output_view.yview)
        out_scroll.pack(side="right", fill="y")
        self.output_view.config(yscrollcommand=out_scroll.set)

        ctrl = tk.Frame(left)
        ctrl.pack(pady=6, anchor="w")
        tk.Button(ctrl, text="Reset",     command=self.reset).pack(
            side="left", padx=2)
        tk.Button(ctrl, text="Go",        command=self.go).pack(
            side="left", padx=2)
        tk.Button(ctrl, text="Stop",      command=self.stop).pack(
            side="left", padx=2)
        tk.Button(ctrl, text="Slower",    command=self.slower).pack(
            side="left", padx=(12, 2))
        tk.Button(ctrl, text="Faster",    command=self.faster).pack(
            side="left", padx=2)
        tk.Button(ctrl, text="Step Back", command=self.step_back).pack(
            side="left", padx=(12, 2))
        tk.Button(ctrl, text="Step Fwd",  command=self.step).pack(
            side="left", padx=2)
        self.speed_label = tk.Label(ctrl, text="", font=("Sans", 10), width=16)
        self.speed_label.pack(side="left", padx=4)

        self._update_speed_label()
        if src:
            self.editor_grid.load_src(src)
        self.reset()

    def load_file(self):
        """
        Prompt for a .bf file, load it into the editor, and reset
        """
        path = filedialog.askopenfilename(
            filetypes=[("Befunge", "*.bf"), ("All files", "*.*")])
        if not path:
            return
        with open(path) as f:
            src = f.read()
        self.editor_grid.load_src(src)
        self.reset()

    def clear(self):
        """
        Empty the editor grid and reset execution state
        """
        self.editor_grid.load_src("")
        self.reset()

    def show_graph(self):
        """
        Build the current program's execution digraph (viz/graph.py) and open
        it as an interactive HTML page in the browser; on an unsupported
        program (string mode) report it in the status line instead
        """
        import subprocess
        import webbrowser
        try:
            nodes = graph.build_graph(self.editor_grid.char_grid())
        except graph.Unsupported as e:
            self.status.config(text=f"graph: {e}")
            return
        if not nodes:
            self.status.config(text="graph: empty program")
            return
        # write inside the repo, not /tmp: sandboxed browsers (snap/flatpak)
        # can't read /tmp but can read non-hidden files under home
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.join(root, "befunge_graph.html")
        with open(out, "w") as f:
            f.write(graph.to_html(nodes))
        url = "file://" + out
        try:                          # detached so the browser doesn't block us
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            webbrowser.open(url)
        self.status.config(text=f"graph: wrote {out}")

    def reset(self):
        """
        Reload the interpreter from the editor and clear
        history/highlights
        """
        self.running = False
        src = self.editor_grid.dump_src()
        self.interp.load(src)
        self.history.clear()
        self.refresh()

    def step(self):
        """
        Push an undo snapshot, advance one instruction, and refresh
        """
        if self.interp.halted:
            return
        self.history.append(self.interp.snapshot())
        if len(self.history) > 1000:
            self.history.pop(0)
        self.interp.step()
        self.refresh()

    def step_back(self):
        """
        Restore the most recent undo snapshot and refresh
        """
        self.running = False
        if not self.history:
            return
        self.interp.restore(self.history.pop())
        self.refresh()

    def go(self):
        """
        Start continuous execution via the tick loop
        """
        if self.interp.halted:
            return
        self.running = True
        self._tick()

    def stop(self):
        """
        Pause continuous execution
        """
        self.running = False

    def slower(self):
        """
        Halve the step rate (fewer steps per tick, then longer delay)
        """
        if self.steps_per_tick > 1:
            self.steps_per_tick //= 2
        else:
            self.delay = min(2000, self.delay * 2)
        self._update_speed_label()

    def faster(self):
        """
        Double the step rate (shorter delay, then more steps per tick)
        """
        if self.delay > 1:
            self.delay = max(1, self.delay // 2)
        else:
            self.steps_per_tick = min(100000, self.steps_per_tick * 2)
        self._update_speed_label()

    def _update_speed_label(self):
        """
        Refresh the speed label with the current steps-per-second rate
        """
        rate = int(self.steps_per_tick * 1000 / self.delay)
        self.speed_label.config(text=f"{rate:,} steps/s")

    def _tick(self):
        """
        Run one batch of steps, refresh, and reschedule while running
        """
        if not self.running or self.interp.halted:
            return
        for _ in range(self.steps_per_tick):
            self.history.append(self.interp.snapshot())
            if len(self.history) > 1000:
                self.history.pop(0)
            self.interp.step()
            if self.interp.halted:
                break
        self.refresh()
        if not self.interp.halted:
            self.root.after(self.delay, self._tick)

    def refresh(self):
        """
        Syncs every view to the current interpreter state.

        Redraws the display grid and IP, and rebuilds the status line, stack,
        data-register, and output views.

        Returns
        -------
        None
        """
        self.display_grid.update_from_array(self.interp.grid)
        self.display_grid.highlight_ip(self.interp.x, self.interp.y)

        arrow = {(1,0): ">", (-1,0): "<", (0,1): "v", (0,-1): "^"}.get(
            (self.interp.dx, self.interp.dy), "?")
        mode = "STRING" if self.interp.string_mode else "normal"
        if self.interp.error:
            tail = f" [ERROR: {self.interp.error}]"
        elif self.interp.halted:
            tail = " [HALTED]"
        else:
            tail = ""
        self.status.config(
            text=(f"IP: ({self.interp.x}, {self.interp.y}) {arrow}   "
                  f"mode: {mode}{tail}"))

        self.stack_view.configure(state="normal")
        self.stack_view.delete("1.0", "end")
        self.stack_view.insert(
            "1.0", " ".join(str(v) for v in self.interp.stack))
        self.stack_view.configure(state="disabled")

        self.output_view.configure(state="normal")
        self.output_view.delete("1.0", "end")
        self.output_view.insert("1.0", self.interp.output)
        self.output_view.see("end")
        self.output_view.configure(state="disabled")

        # data registers: list the nonzero ones (sparse tape)
        self.reg_view.configure(state="normal")
        self.reg_view.delete("1.0", "end")
        regs = self.interp.regs
        if not regs:
            self.reg_view.insert("1.0", "(empty)")
        else:
            for i in sorted(regs):
                self.reg_view.insert("end", f"r{i}={int(regs[i])}  ")
        self.reg_view.configure(state="disabled")

    def run(self):
        """
        Enter the tkinter main loop
        """
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
