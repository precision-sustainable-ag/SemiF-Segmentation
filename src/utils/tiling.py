def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering [0, length) with the given stride; the last tile is
    aligned to the end instead of being dropped or running past it."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts
