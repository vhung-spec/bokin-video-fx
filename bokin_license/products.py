"""The Bokin products a key can unlock: one bit each in the key's 32-bit product mask.

A key lists one product or several (a bundle). Never renumber or reuse a bit: keys already issued
carry these numbers. New products take the next free bit.
"""
from __future__ import annotations

PRODUCTS = {
    # id: (bit, display name)
    "studio-music": (0, "Bokin Studio Music"),
    "playlist-studio": (1, "Bokin Playlist Studio"),
    "fx": (2, "Bokin FX"),
    "loop": (3, "Bokin Loop"),
    "motion-studio": (4, "Bokin Motion Studio"),
    "upscaler": (5, "Bokin Upscaler"),
}


class UnknownProduct(ValueError):
    pass


def bit(product):
    try:
        return PRODUCTS[product][0]
    except KeyError:
        raise UnknownProduct(f"Sản phẩm không có trong danh sách: {product!r}") from None


def mask(products):
    """32-bit mask of product ids (at least one)."""
    products = list(products)
    if not products:
        raise UnknownProduct("Hãy chọn ít nhất một sản phẩm.")
    value = 0
    for product in products:
        value |= 1 << bit(product)
    return value


def products_of(value):
    """Known product ids in a mask, in the registry's order (unknown bits are ignored)."""
    return [product for product, (index, _) in PRODUCTS.items() if value >> index & 1]


def display_name(product):
    return PRODUCTS[product][1] if product in PRODUCTS else product


def display_names(products):
    return ", ".join(display_name(product) for product in products)
