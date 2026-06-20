import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pdal

from opendm.dem.commands import compute_outlier_safe_bounds


def write_las(path, x, y, z):
    arr = np.empty(len(x), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    arr["X"], arr["Y"], arr["Z"] = x, y, z
    pdal.Pipeline(json.dumps([{"type": "writers.las", "filename": path}]), arrays=[arr]).execute()


class TestOutlierSafeBounds(unittest.TestCase):
    """compute_outlier_safe_bounds only crops when a few gross outliers have
    inflated the extent, and never touches well-behaved point clouds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _las(self, name, x, y, z):
        path = os.path.join(self.tmp, name)
        write_las(path, x, y, z)
        return path

    def test_keeps_normal_small_scene(self):
        # A dense 80 x 80 m scene: raster is small, nothing to do.
        n = 100000
        x = self.rng.uniform(0, 80, n)
        y = self.rng.uniform(0, 80, n)
        z = self.rng.normal(0, 1.5, n)
        self.assertIsNone(compute_outlier_safe_bounds(self._las("small.las", x, y, z), 0.05))

    def test_keeps_large_uniform_survey(self):
        # A genuine 800 x 800 m survey (256 MP raster) with no outliers: the
        # extent is large but not inflated, so it must be left untouched.
        n = 200000
        x = self.rng.uniform(0, 800, n)
        y = self.rng.uniform(0, 800, n)
        z = self.rng.normal(0, 2, n)
        self.assertIsNone(compute_outlier_safe_bounds(self._las("large.las", x, y, z), 0.05))

    def test_keeps_thin_transect(self):
        # A long, thin corridor (2000 x 40 m): high aspect ratio, no outliers.
        n = 200000
        x = self.rng.uniform(0, 2000, n)
        y = self.rng.uniform(0, 40, n)
        z = self.rng.normal(0, 2, n)
        self.assertIsNone(compute_outlier_safe_bounds(self._las("transect.las", x, y, z), 0.05))

    def test_crops_gross_outliers(self):
        # A dense 80 x 80 m scene plus a handful of flyers up to 3 km away.
        n, f = 200000, 30
        x = np.concatenate([self.rng.uniform(0, 80, n), self.rng.uniform(-3000, 3000, f)])
        y = np.concatenate([self.rng.uniform(0, 80, n), self.rng.uniform(-3000, 3000, f)])
        z = np.concatenate([self.rng.normal(0, 1.5, n), self.rng.uniform(-3000, 3000, f)])
        bounds = compute_outlier_safe_bounds(self._las("flyers.las", x, y, z), 0.05)
        self.assertIsNotNone(bounds)
        minx, miny, maxx, maxy = bounds
        # The robust window should hug the real ~80 m scene, not the km-wide extent.
        self.assertLess(maxx - minx, 200)
        self.assertLess(maxy - miny, 200)


if __name__ == "__main__":
    unittest.main()
