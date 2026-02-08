import numpy as np
from collections import deque

from src.cooling_network.types import CellType


class CoolingNetwork:
    def __init__(self, C, layer = 1):
        self.C = C
        self.layer = layer

    def clone(self) -> "CoolingNetwork":
        return CoolingNetwork(C=self.C.copy(), layer=self.layer)

    def in_bounds(self, i: int, j: int) -> bool:
        n, m = self.C.shape
        return 0 <= i < n and 0 <= j < m

    def neighbors4(self, i: int, j: int):
        cand = ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
        return [(x, y) for (x, y) in cand if self.in_bounds(x, y)]

    def is_active(self, i: int, j: int) -> bool:
        return self.C[i, j] in (CellType.LIQUID, CellType.INLET, CellType.OUTLET)

    def remove_cell(self, i: int, j: int) -> None:
        tag = self.C[i, j]
        if tag in (CellType.TSV, CellType.INLET, CellType.OUTLET):
            raise ValueError(f"Cannot remove cell {tag} at ({i}, {j})")
        if tag != CellType.LIQUID:
            raise ValueError(f"Cell {tag} at ({i}, {j}) is not liquid")

        self.C[i, j] = CellType.SILICON

    def find_inlets_outlets(self):
        inlets = []
        outlets = []
        n, m = self.C.shape
        for i in range(n):
            for j in range(m):
                if self.C[i, j] == CellType.INLET:
                    inlets.append((i, j))
                elif self.C[i, j] == CellType.OUTLET:
                    outlets.append((i, j))
        return inlets, outlets


    def has_inlet_to_outlet_path(self) -> bool:
        inlets, outlets = self.find_inlets_outlets()

        if not inlets or not outlets:
            return False

        q = deque(inlets)
        visited = set(inlets)

        while q:
            i, j = q.popleft()
            if self.C[i, j] == CellType.OUTLET:
                return True

            for ni, nj in self.neighbors4(i, j):
                if (ni, nj) in visited:
                    continue
                if not self.is_active(ni, nj):
                    continue
                visited.add((ni, nj))
                q.append((ni, nj))

        return False

    def prune_irregular(self) -> int:
        """
        Removes:
        1) components with no inlet or no outlet (isolated/one-sided),
        2) dead-end liquid branches (degree<=1 peeling, terminals protected).
        Returns number of LIQUID cells removed.
        """
        removed = 0
        n, m = self.C.shape

        # ---------- Step A: remove components without (inlet and outlet) ----------
        active = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                active[i, j] = self.is_active(i, j)

        visited = np.zeros((n, m), dtype=bool)

        for si in range(n):
            for sj in range(m):
                if not active[si, sj] or visited[si, sj]:
                    continue

                q = deque([(si, sj)])
                visited[si, sj] = True

                comp = []
                has_inlet = False
                has_outlet = False

                while q:
                    i, j = q.popleft()
                    comp.append((i, j))
                    if self.C[i, j] == CellType.INLET:
                        has_inlet = True
                    elif self.C[i, j] == CellType.OUTLET:
                        has_outlet = True

                    for ni, nj in self.neighbors4(i, j):
                        if active[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            q.append((ni, nj))

                # якщо компонент не має хоча б одного inlet і одного outlet -> чистимо LIQUID
                if not (has_inlet and has_outlet):
                    for i, j in comp:
                        if self.C[i, j] == CellType.LIQUID:
                            self.C[i, j] = CellType.SILICON
                            removed += 1
                            active[i, j] = False  # обновляємо локально

        # Перерахувати active після масового видалення
        for i in range(n):
            for j in range(m):
                active[i, j] = self.is_active(i, j)

        # ---------- Step B: peel dead-ends (LIQUID with degree <= 1) ----------
        deg = np.zeros((n, m), dtype=np.int16)
        for i in range(n):
            for j in range(m):
                if not active[i, j]:
                    continue
                deg[i, j] = sum(1 for (ni, nj) in self.neighbors4(i, j) if active[ni, nj])

        q = deque()
        for i in range(n):
            for j in range(m):
                if self.C[i, j] == CellType.LIQUID and deg[i, j] <= 1:
                    q.append((i, j))

        while q:
            i, j = q.popleft()
            if self.C[i, j] != CellType.LIQUID:
                continue
            if deg[i, j] > 1:
                continue

            # remove it
            self.C[i, j] = CellType.SILICON
            removed += 1
            active[i, j] = False

            # update neighbors degrees
            for ni, nj in self.neighbors4(i, j):
                if not active[ni, nj]:
                    continue
                deg[ni, nj] -= 1
                if self.C[ni, nj] == CellType.LIQUID and deg[ni, nj] <= 1:
                    q.append((ni, nj))

            deg[i, j] = 0

        return removed