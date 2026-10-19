from __future__ import annotations

import unittest

from idphoto.config import (
    ONE_INCH,
    TWO_INCH,
    IDPhotoSize,
    custom_size,
    parse_rgb,
)


class SizeTest(unittest.TestCase):
    def test_standard_pixel_dimensions(self) -> None:
        self.assertEqual(ONE_INCH.pixels, (295, 413))
        self.assertEqual(TWO_INCH.pixels, (413, 579))

    def test_pixel_dimensions_keep_five_by_seven_ratio(self) -> None:
        self.assertAlmostEqual(ONE_INCH.aspect_ratio, 5 / 7, delta=0.002)
        self.assertAlmostEqual(TWO_INCH.aspect_ratio, 5 / 7, delta=0.002)

    def test_from_mm_converts_at_given_dpi(self) -> None:
        size = IDPhotoSize.from_mm("square", "方图", 25.4, 50.8, dpi=200)
        self.assertEqual(size.pixels, (200, 400))
        self.assertEqual(size.dpi, 200)

    def test_custom_size(self) -> None:
        size = custom_size(600, 800, dpi=300)
        self.assertEqual(size.pixels, (600, 800))
        self.assertEqual(size.key, "custom")

    def test_custom_size_rejects_non_positive(self) -> None:
        with self.assertRaises(ValueError):
            custom_size(0, 800)
        with self.assertRaises(ValueError):
            custom_size(600, -1)


class ParseRgbTest(unittest.TestCase):
    def test_comma_format(self) -> None:
        self.assertEqual(parse_rgb("67,142,219"), (67, 142, 219))
        self.assertEqual(parse_rgb(" 255 , 0 , 0 "), (255, 0, 0))

    def test_hex_format(self) -> None:
        self.assertEqual(parse_rgb("#438EDB"), (67, 142, 219))
        self.assertEqual(parse_rgb("438edb"), (67, 142, 219))

    def test_invalid_formats(self) -> None:
        for value in ("", "abc", "1,2", "1,2,3,4", "#GGGGGG"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_rgb(value)

    def test_out_of_range(self) -> None:
        for value in ("256,0,0", "-1,0,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_rgb(value)


if __name__ == "__main__":
    unittest.main()
