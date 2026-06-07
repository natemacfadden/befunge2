# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Turn a Befunge-93 program into an execution digraph. Nodes are
#               the non-space cells (the chars that get executed); edges follow
#               the instruction pointer to the next node, keyed by the heading
#               the IP arrives with and the heading it leaves with. Spaces are
#               not nodes - an edge jumps straight to the next non-space cell.
# -----------------------------------------------------------------------------

import html
import json
import math
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import befunge as bf

# a heading is the direction the IP travels, with its (dx, dy) grid step
DELTA = {'right': (1, 0), 'left': (-1, 0), 'up': (0, -1), 'down': (0, 1)}
HEADINGS = tuple(DELTA)   # the four heading names, in DELTA's order

# the two branches out of a conditional (_ |): the popped value zero or nonzero
CONDS = ('zero', 'nonzero')

# Colors/dashes per branch kind, shared by python (markers) and js. 'start' is
# the synthetic, off-playfield entry node (drawn green).
_COLOR = {'fixed': '#1f77b4', 'pass': '#7f7f7f', 'stack': '#ff7f0e',
          'random': '#9467bd', 'halt': '#d62728', 'start': '#2ca02c'}
_DASH = {'stack': '6,4', 'random': '2,4'}

# the html viewer template (placeholders filled in by to_html)
_HTML = (Path(__file__).parent / 'graph_template.html').read_text()


class Unsupported(ValueError):
    """
    Raised for a program outside the subset the digraph models: no string
    mode (the digraph assumes every cell executes as its instruction).
    """


# =============================================================================
# Graph construction
# =============================================================================

def transition(ch, arrival_heading, zero):
    """
    Resolve where the IP goes after executing a cell.

    Parameters
    ----------
    ch : str
        The character executed at the cell.
    arrival_heading : str
        The heading the IP enters the cell with (one of HEADINGS).
    zero : bool
        Whether the value the IP would pop is zero; only the conditionals
        (_ |) read it.

    Returns
    -------
    dist : dict
        The exit heading(s) the IP can leave by (empty for '@').
    kind : str
        What drives the exit:
          'fixed'  - redirect by the char (> < ^ v)
          'stack'  - set by the popped value (_ |)
          'random' - ?
          'halt'   - @ (no successor)
          'pass'   - inertia: keep the arrival heading (everything else)
    """
    if ch == '>': return {'right': 1.0}, 'fixed'
    if ch == '<': return {'left': 1.0}, 'fixed'
    if ch == '^': return {'up': 1.0}, 'fixed'
    if ch == 'v': return {'down': 1.0}, 'fixed'
    if ch == '_': return {'right' if zero else 'left': 1.0}, 'stack'
    if ch == '|': return {'down' if zero else 'up': 1.0}, 'stack'
    if ch == '?': return {h: 0.25 for h in HEADINGS}, 'random'
    if ch == '@': return {}, 'halt'
    return {arrival_heading: 1.0}, 'pass'


def next_node(grid, x, y, heading, skip=0):
    """
    Walk from a cell along a heading to the next node.

    Follows the heading across the W x H torus and returns the first non-space
    cell - the node this edge points at.

    Parameters
    ----------
    grid : list of list of str
        The (H, W) char grid.
    x, y : int
        The starting cell.
    heading : str
        Direction to walk (one of HEADINGS).
    skip : int, optional
        Cells to skip before looking; `#` (bridge) uses skip=1 to jump the
        cell after it. Defaults to 0.

    Returns
    -------
    tuple or None
        (x, y) of the next non-space cell, or None if the whole ray is blank.
    """
    dx, dy = DELTA[heading]
    cx, cy = x, y
    for _ in range(bf.W * bf.H):
        cx = (cx + dx) % bf.W
        cy = (cy + dy) % bf.H
        if skip > 0:
            skip -= 1
            continue
        if grid[cy][cx] != ' ':
            return (cx, cy)
    return None


def build_graph(grid, verify=True):
    """
    Build the execution digraph from a char grid.

    Parameters
    ----------
    grid : list of list of str
        The (H, W) char grid.
    verify : bool, optional
        Raise Unsupported for programs outside the subset. Defaults to True.

    Returns
    -------
    dict
        {pos: node}, where pos is (x, y) and node has:
          char    - the executed character
          kind    - branch kind from transition()
          targets - {exit_heading: dst_pos or None}, the next node per heading
          trans   - {arrival_heading: {cond: {exit_heading: weight}}}, the full
                    4 x 2 x 4 table (every cell present)
        Plus a synthetic 'start' node at (-1, entry_y) marking the origin.
    """
    # The digraph assumes every cell executes as its instruction, so string
    # mode (") is outside the supported subset.
    if verify and any('"' in row for row in grid):
        raise Unsupported('string mode (") is not in the supported subset')
    nodes = {}
    for y in range(bf.H):
        for x in range(bf.W):
            ch = grid[y][x]
            if ch == ' ':
                continue
            skip = 1 if ch == '#' else 0
            targets = {h: next_node(grid, x, y, h, skip) for h in HEADINGS}
            kind = transition(ch, 'right', True)[1]
            trans = {}
            for arrival_heading in HEADINGS:
                trans[arrival_heading] = {}
                for cond in CONDS:
                    dist, _ = transition(ch, arrival_heading, cond == 'zero')
                    trans[arrival_heading][cond] = {
                        h: dist.get(h, 0.0) for h in HEADINGS}
            nodes[(x, y)] = {'pos': (x, y), 'char': ch, 'kind': kind,
                             'targets': targets, 'trans': trans}
    # The first executed cell: (0,0) if non-blank, else the first non-space
    # cell to its right (the IP starts at (0,0) heading right).
    e = (0, 0) if grid[0][0] != ' ' else next_node(grid, 0, 0, 'right')
    if e is not None and e in nodes:
        # Mark it, and add a synthetic ">" start node one column to its left
        # with a single edge in. It has no incoming edges (the IP can never
        # re-enter it), so it just marks the execution origin in the graph.
        nodes[e]['entry'] = True
        nodes[(-1, e[1])] = {'pos': (-1, e[1]), 'char': '>', 'kind': 'start',
                             'target': e}
    return nodes


# =============================================================================
# Rendering
# =============================================================================

def edges(nodes):
    """
    Yield (src, dst, arrival_heading, cond, exit_heading, prob, kind) for
    every nonzero transition that lands on a real node.
    """
    for pos, node in nodes.items():
        if node['kind'] == 'start':                 # synthetic entry edge
            yield pos, node['target'], 'start', '-', 'right', 1.0, 'start'
            continue
        for arrival_heading in HEADINGS:
            for cond in CONDS:
                dist = node['trans'][arrival_heading][cond]
                for exit_heading, p in dist.items():
                    dst = node['targets'][exit_heading]
                    if p > 0 and dst is not None:
                        yield (pos, dst, arrival_heading, cond,
                               exit_heading, p, node['kind'])


def to_dot(nodes):
    """
    Render the digraph as graphviz DOT. Stack branches are dashed, random
    dotted, everything else solid; edge labels read "<arrival>><exit> <prob>".
    """
    def esc(c):
        return c.replace('\\', '\\\\').replace('"', '\\"')

    out = ['digraph bf {', '  node [shape=box fontname=monospace];']
    for (x, y), node in nodes.items():
        out.append(f'  "{x},{y}" [label="{esc(node["char"])}\\n({x},{y})"];')
    for s, d, arrival_heading, cond, exit_heading, p, kind in edges(nodes):
        style = {'stack': 'dashed', 'random': 'dotted'}.get(kind, 'solid')
        # The label reads "<arrival>/<cond>><exit> <prob>"; cond is shown only
        # when it changes the outcome (stack branches).
        tag = (f'{arrival_heading[0]}/{cond[0]}' if kind == 'stack'
               else arrival_heading[0])
        out.append(f'  "{s[0]},{s[1]}" -> "{d[0]},{d[1]}" '
                   f'[label="{tag}>{exit_heading[0]} {p:g}", style={style}];')
    out.append('}')
    return '\n'.join(out)


def dump(nodes):
    """
    Human-readable per-node listing of the 4x4 transition table.
    """
    for pos in sorted(nodes, key=lambda p: (p[1], p[0])):
        node = nodes[pos]
        print(f'({pos[0]:>2},{pos[1]:>2}) {node["char"]!r:4} [{node["kind"]}]')
        if node['kind'] == 'start':
            print(f'    start -> {node["target"]}')
            continue
        # The stack-zero bit only changes the outcome for conditionals, so for
        # everything else collapse the two conds into one line.
        conds = CONDS if node['kind'] == 'stack' else ('zero',)
        targets = node['targets']
        for arrival_heading in HEADINGS:
            for cond in conds:
                dist = node['trans'][arrival_heading][cond]
                moves = [f'{exit_heading}->{targets[exit_heading]} ({p:g})'
                         for exit_heading, p in dist.items()
                         if p > 0 and targets[exit_heading] is not None]
                tag = (f'{arrival_heading:5} {cond:7}'
                       if node['kind'] == 'stack'
                       else f'{arrival_heading:5}        ')
                print(f'    arrives {tag}: ' +
                      (', '.join(moves) if moves else '(halt)'))


def _node_tip(pos, node):
    if node['kind'] == 'start':
        return ('<b>start (synthetic)</b><br>not on the playfield; IP begins '
                f"here heading right &rarr; {node['target']}")
    lines = [f"<b>{html.escape(repr(node['char']))} at ({pos[0]},{pos[1]})</b>",
             f"kind: {node['kind']}"]
    conds = CONDS if node['kind'] == 'stack' else ('zero',)
    for arrival_heading in HEADINGS:
        for cond in conds:
            dist = node['trans'][arrival_heading][cond]
            for exit_heading, p in dist.items():
                tgt = node['targets'][exit_heading]
                if p > 0 and tgt is not None:
                    c = f"/{cond}" if node['kind'] == 'stack' else ""
                    lines.append(f"arrives {arrival_heading}{c} &rarr; "
                                 f"{exit_heading} {tgt} (p={p:g})")
    if node['kind'] == 'halt':
        lines.append("(no exits)")
    return '<br>'.join(lines)


def _edge_tip(s, d, kind, items):
    if kind == 'start':
        return f"<b>start</b><br>program entry &rarr; first executed cell {d}"
    lines = [f"<b>({s[0]},{s[1]}) &rarr; ({d[0]},{d[1]})</b>",
             f"kind: {kind}"]
    for arrival_heading, cond, exit_heading, p in items:
        c = f"/{cond}" if kind == 'stack' else ""
        lines.append(
            f"arrives {arrival_heading}{c} &rarr; exits {exit_heading} "
            f"(p={p:g})")
    return '<br>'.join(lines)


def _force_layout(pts, jedges, side, margin, iters=300):
    """
    Spread nodes out with a deterministic force-directed pass.

    A seeded Fruchterman-Reingold relaxation: nodes repel, edges pull together,
    shortening edges and removing most crossings. No rng, so the layout is
    stable run to run.

    Parameters
    ----------
    pts : list of [x, y]
        Initial node positions (seeded from the grid).
    jedges : list of dict
        Edges with 's'/'d' endpoint indices into pts.
    side : float
        Side length of the square layout area.
    margin : float
        Padding added around the refit result.
    iters : int, optional
        Relaxation iterations. Defaults to 300.

    Returns
    -------
    list of [x, y]
        New positions, refit and centered in the square canvas.
    """
    n = len(pts)
    if n < 2:
        return [list(p) for p in pts]
    p = [list(q) for q in pts]
    adj = {(min(e['s'], e['d']), max(e['s'], e['d']))
           for e in jedges if e['s'] != e['d']}
    k = math.sqrt(side * side / n) * 0.7   # ideal edge length
    t = side * 0.12                        # max step, cooled each iteration
    for _ in range(iters):
        disp = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):                 # repulsion across all pairs
            for j in range(i + 1, n):
                dx, dy = p[i][0] - p[j][0], p[i][1] - p[j][1]
                d = math.hypot(dx, dy) or 0.01
                f = k * k / d
                ux, uy = dx / d, dy / d
                disp[i][0] += ux * f; disp[i][1] += uy * f
                disp[j][0] -= ux * f; disp[j][1] -= uy * f
        for a, b in adj:                   # attraction along edges
            dx, dy = p[a][0] - p[b][0], p[a][1] - p[b][1]
            d = math.hypot(dx, dy) or 0.01
            f = d * d / k
            ux, uy = dx / d, dy / d
            disp[a][0] -= ux * f; disp[a][1] -= uy * f
            disp[b][0] += ux * f; disp[b][1] += uy * f
        for i in range(n):                 # move, capped by temperature
            dx, dy = disp[i]
            d = math.hypot(dx, dy) or 0.01
            m = min(d, t)
            p[i][0] += dx / d * m; p[i][1] += dy / d * m
        t *= 0.97
    xs = [q[0] for q in p]; ys = [q[1] for q in p]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = (maxx - minx) or 1, (maxy - miny) or 1
    scale = min(side / w, side / h)
    ox = margin + (side - w * scale) / 2
    oy = margin + (side - h * scale) / 2
    return [[ox + (q[0] - minx) * scale, oy + (q[1] - miny) * scale] for q in p]


def to_html(nodes):
    """
    Render the digraph as a standalone, self-contained HTML page.

    Inline SVG plus vanilla JS, no external deps, opens offline. Nodes start at
    their grid coordinates ("home"); they are draggable, and the "reset layout"
    button snaps them back home. Parallel transitions between the same two
    cells are merged into one drawn edge whose tooltip lists every
    (arrival, stack-cond) that takes it, which keeps overlap manageable.

    Torus-wrap edges are drawn as straight bows across the grid, so an edge
    labelled 'left' can visually point right -- trust the hover, not the
    geometry.

    Parameters
    ----------
    nodes : dict
        The digraph from build_graph().

    Returns
    -------
    str
        A complete HTML document.
    """
    cell, r, margin, minside = 80, 16, 60, 1100
    # an empty program has no nodes; fall back to a single cell so the layout
    # math below stays well-defined and the page renders blank
    xs = [x for x, _ in nodes] or [0]
    ys = [y for _, y in nodes] or [0]
    minx, miny = min(xs), min(ys)
    # Square canvas (at least minside) with the grid layout centered in it, so
    # wide one-liners and tall programs both get room to be dragged around.
    cw, ch = (max(xs) - minx) * cell, (max(ys) - miny) * cell
    side = max(cw, ch, minside)
    total = side + 2 * margin
    offx, offy = margin + (side - cw) / 2, margin + (side - ch) / 2
    wpx = hpx = total

    idx = {pos: i for i, pos in enumerate(nodes)}
    jnodes, grid_pts = [], []
    for pos, node in nodes.items():
        grid_pts.append([offx + (pos[0] - minx) * cell,
                         offy + (pos[1] - miny) * cell])
        jnodes.append({'char': node['char'], 'kind': node['kind'],
                       'tip': _node_tip(pos, node)})

    agg = {}
    for s, d, arrival_heading, cond, exit_heading, p, kind in edges(nodes):
        agg.setdefault((s, d), {'kind': kind, 'items': []})['items'].append(
            (arrival_heading, cond, exit_heading, p))
    jedges = [{'s': idx[s], 'd': idx[d], 'kind': info['kind'],
               'tip': _edge_tip(s, d, info['kind'], info['items'])}
              for (s, d), info in agg.items()]

    # two home layouts: 'grid' mirrors the source, 'tidy' minimizes crossings
    tidy_pts = _force_layout(grid_pts, jedges, side, margin)

    markers = ''.join(
        f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for k, c in _COLOR.items())

    # default to the grid layout when it's known, else the force layout
    layouts = {'grid': grid_pts, 'tidy': tidy_pts}
    data = {'W': wpx, 'H': hpx, 'R': r, 'colors': _COLOR, 'dash': _DASH,
            'nodes': jnodes, 'edges': jedges,
            'start': 'grid' if 'grid' in layouts else 'tidy',
            'layouts': layouts}
    return (_HTML.replace('__W__', str(wpx)).replace('__H__', str(hpx))
            .replace('__MARKERS__', markers)
            .replace('__DATA__', json.dumps(data)))


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit('usage: python viz/graph.py <file.bf> '
                 '[--dump | --dot] [--html OUT] [--no-open]')
    with open(args[0]) as f:
        src = f.read()
    # lay the source onto an H x W char grid, space-padded and edge-clipped
    grid = [[' '] * bf.W for _ in range(bf.H)]
    for y, line in enumerate(src.split('\n')[:bf.H]):
        for x, ch in enumerate(line[:bf.W]):
            grid[y][x] = ch
    try:
        g = build_graph(grid)
    except Unsupported as e:
        sys.exit(f'rejected: {e}')
    if '--dot' in args:
        print(to_dot(g))
    elif '--dump' in args:
        dump(g)
    else:
        # default action: write the html viewer next to the source and open it
        out = os.path.splitext(args[0])[0] + '.html'
        if '--html' in args:                  # optional explicit output path
            i = args.index('--html')
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                out = args[i + 1]
        with open(out, 'w') as f:
            f.write(to_html(g))
        print(f'wrote {out}')
        if '--no-open' not in args:
            # Launch detached with output swallowed: otherwise the browser
            # inherits the terminal (gtk noise leaks) and python blocks on it.
            url = 'file://' + os.path.abspath(out)
            try:
                subprocess.Popen(['xdg-open', url], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
            except OSError:
                webbrowser.open(url)
