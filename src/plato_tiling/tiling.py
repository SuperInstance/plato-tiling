"""Tiling — smart content decomposition with semantic boundaries and adaptive chunking."""
import re
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class TileType(Enum):
    FACT = "fact"
    OPINION = "opinion"
    QUESTION = "question"
    COMMAND = "command"
    NARRATIVE = "narrative"
    CODE = "code"
    METADATA = "metadata"
    MIXED = "mixed"

class BoundaryHint(Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"
    LINE_BREAK = "line_break"

@dataclass
class Tile:
    content: str
    tile_type: TileType = TileType.MIXED
    confidence: float = 0.5
    index: int = 0
    boundary_hint: BoundaryHint = BoundaryHint.PARAGRAPH
    source_offset: int = 0
    source_length: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class TilingConfig:
    max_tile_size: int = 500
    min_tile_size: int = 30
    overlap: int = 0
    preserve_code_blocks: bool = True
    detect_types: bool = True
    adaptive: bool = True

class Tiler:
    def __init__(self, config: TilingConfig = None):
        self.config = config or TilingConfig()

    def tile(self, content: str, tile_type: str = "") -> list[Tile]:
        if not content or not content.strip():
            return []
        # Detect best strategy
        if self.config.adaptive:
            strategy = self._detect_strategy(content)
        else:
            strategy = "paragraph"
        tiles = self._apply_strategy(content, strategy)
        # Type detection
        if self.config.detect_types:
            for tile in tiles:
                tile.tile_type = self._classify(tile.content)
        # Split oversized tiles
        result = []
        for tile in tiles:
            if len(tile.content) > self.config.max_tile_size:
                result.extend(self._split_large(tile))
            elif len(tile.content) < self.config.min_tile_size:
                if result:
                    result[-1].content += "\n" + tile.content
                else:
                    result.append(tile)
            else:
                result.append(tile)
        # Re-index
        for i, t in enumerate(result):
            t.index = i
        return result

    def _detect_strategy(self, content: str) -> str:
        has_headings = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
        has_code = "```" in content
        has_lists = bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE))
        has_numbered_lists = bool(re.search(r'^\s*\d+\.\s', content, re.MULTILINE))
        if has_code and len(content) > self.config.max_tile_size:
            return "code_aware"
        if has_headings:
            return "heading"
        if has_lists or has_numbered_lists:
            return "list_aware"
        return "paragraph"

    def _apply_strategy(self, content: str, strategy: str) -> list[Tile]:
        tiles = []
        if strategy == "heading":
            sections = re.split(r'\n(?=#{1,6}\s)', content.strip())
            for i, section in enumerate(sections):
                if section.strip():
                    tiles.append(Tile(content=section.strip(), boundary_hint=BoundaryHint.HEADING))
        elif strategy == "code_aware":
            parts = re.split(r'(```\w*\n.*?```)', content, flags=re.DOTALL)
            for i, part in enumerate(parts):
                if part.strip():
                    hint = BoundaryHint.CODE_BLOCK if part.startswith("```") else BoundaryHint.PARAGRAPH
                    tiles.append(Tile(content=part.strip(), boundary_hint=hint))
        elif strategy == "list_aware":
            items = re.split(r'\n(?=\s*[-*+]\s|\s*\d+\.\s)', content.strip())
            for item in items:
                if item.strip():
                    tiles.append(Tile(content=item.strip(), boundary_hint=BoundaryHint.LIST_ITEM))
        else:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for p in paragraphs:
                tiles.append(Tile(content=p, boundary_hint=BoundaryHint.PARAGRAPH))
        return tiles

    def _split_large(self, tile: Tile) -> list[Tile]:
        content = tile.content
        max_size = self.config.max_tile_size
        overlap = self.config.overlap
        chunks = []
        if self.config.preserve_code_blocks and "```" in content:
            # Split at code block boundaries first
            parts = re.split(r'(```)', content)
            current = ""
            for part in parts:
                if len(current + part) > max_size and current:
                    chunks.append(current)
                    if overlap:
                        current = current[-overlap:] + part
                    else:
                        current = part
                else:
                    current += part
            if current.strip():
                chunks.append(current)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current = ""
            for sent in sentences:
                if len(current + " " + sent) > max_size and current:
                    chunks.append(current.strip())
                    if overlap:
                        words = current.split()
                        current = " ".join(words[-overlap:]) + " " + sent
                    else:
                        current = sent
                else:
                    current = (current + " " + sent).strip()
            if current.strip():
                chunks.append(current)
        return [Tile(content=c, tile_type=tile.tile_type, boundary_hint=BoundaryHint.SENTENCE,
                    source_offset=tile.source_offset, tags=tile.tags.copy()) for c in chunks if c.strip()]

    def _classify(self, content: str) -> TileType:
        stripped = content.strip()
        if stripped.startswith("```") or stripped.startswith("def ") or stripped.startswith("class "):
            return TileType.CODE
        if stripped.endswith("?"):
            return TileType.QUESTION
        if re.match(r'^[A-Z]', stripped) and any(w in stripped.lower() for w in ("i think", "i believe", "in my opinion", "should", "could")):
            return TileType.OPINION
        if re.match(r'^\s*[-*+]\s', stripped) or re.match(r'^\s*\d+\.\s', stripped):
            return TileType.FACT
        if any(w in stripped.lower().split() for w in ("please", "do this", "make sure", "ensure", "run")):
            return TileType.COMMAND
        if len(stripped) < 50 and ("=" in stripped or ":" in stripped):
            return TileType.METADATA
        if len(stripped) > 200:
            return TileType.NARRATIVE
        return TileType.MIXED

    def tile_with_overlap(self, content: str, chunk_size: int = 200,
                          overlap_size: int = 50) -> list[Tile]:
        tiles = []
        step = chunk_size - overlap_size
        words = content.split()
        for i in range(0, max(1, len(words) - overlap_size), step):
            chunk = " ".join(words[i:i + chunk_size])
            tiles.append(Tile(content=chunk, source_offset=i,
                            source_length=min(chunk_size, len(words) - i)))
        return tiles

    def estimate_tiles(self, content: str) -> int:
        return max(1, len(content) // self.config.max_tile_size)

    @property
    def stats(self) -> dict:
        return {"config": {"max_size": self.config.max_tile_size,
                "min_size": self.config.min_tile_size,
                "overlap": self.config.overlap,
                "adaptive": self.config.adaptive}}
