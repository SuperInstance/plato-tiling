"""Core tiling engine with ghost lifecycle and adaptive search."""

import time, uuid
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GhostTile:
    original_tile_id: str
    decayed_at: float = field(default_factory=time.time)
    attention_weight: float = 0.1

    def compute_attention(self, active_tiles: list) -> float:
        if not active_tiles: return self.attention_weight
        min_dist = min(abs(t.health - 0.05) for t in active_tiles if hasattr(t, 'health'))
        return self.attention_weight * (1.0 / (1.0 + min_dist))

@dataclass
class Tile:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    domain: str = "general"
    confidence: float = 0.5
    priority: str = "P2"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    health: float = 1.0
    ghost: bool = False
    use_count: int = 0

class TilingEngine:
    def __init__(self, decay_rate: float = 0.99, ghost_threshold: float = 0.05):
        self.decay_rate = decay_rate
        self.ghost_threshold = ghost_threshold
        self._tiles: dict[str, Tile] = {}
        self._ghosts: list[GhostTile] = []

    def create(self, content: str, domain: str = "general", confidence: float = 0.5,
               priority: str = "P2", tags: list[str] = None) -> Tile:
        tile = Tile(content=content, domain=domain, confidence=confidence,
                    priority=priority, tags=tags or [])
        self._tiles[tile.id] = tile
        return tile

    def access(self, tile_id: str) -> Optional[Tile]:
        tile = self._tiles.get(tile_id)
        if tile and not tile.ghost:
            tile.use_count += 1
            tile.health = min(tile.health + 0.1, 1.0)
        return tile

    def ghost_tile(self, tile_id: str) -> Optional[GhostTile]:
        tile = self._tiles.get(tile_id)
        if tile and tile.health < self.ghost_threshold and not tile.ghost:
            tile.ghost = True
            ghost = GhostTile(original_tile_id=tile_id)
            self._ghosts.append(ghost)
            return ghost
        return None

    def resurrect(self, tile_id: str, boost: float = 0.5) -> Optional[Tile]:
        tile = self._tiles.get(tile_id)
        if tile and tile.ghost:
            tile.ghost = False
            tile.health = min(tile.health + boost, 1.0)
            return tile
        return None

    def search(self, query: str, top_n: int = 5) -> list[Tile]:
        q_words = set(query.lower().split())
        results = []
        for t in self._tiles.values():
            if t.ghost: continue
            c_words = set(t.content.lower().split())
            if q_words and c_words:
                overlap = len(q_words & c_words) / len(q_words | c_words)
                results.append((overlap, t))
        results.sort(key=lambda x: -x[0])
        return [t for _, t in results[:top_n]]

    def decay_all(self, rate: float = None):
        r = rate or self.decay_rate
        for t in self._tiles.values():
            if not t.ghost:
                t.health *= r

    def get(self, tile_id: str) -> Optional[Tile]:
        return self._tiles.get(tile_id)

    @property
    def stats(self) -> dict:
        active = [t for t in self._tiles.values() if not t.ghost]
        ghosted = [t for t in self._tiles.values() if t.ghost]
        domains = {}
        for t in self._tiles.values():
            domains[t.domain] = domains.get(t.domain, 0) + 1
        return {"total": len(self._tiles), "active": len(active), "ghosted": len(ghosted),
                "avg_health": sum(t.health for t in active) / max(len(active), 1),
                "domains": domains, "ghost_records": len(self._ghosts)}
