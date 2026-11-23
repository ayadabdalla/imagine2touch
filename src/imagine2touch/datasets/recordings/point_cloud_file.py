"""Point clouds on disk, in the PCD format of the scans."""

import numpy as np


class PointCloudFile:
    """Binary PCD files whose fields are 4-byte floats starting with x, y, z."""

    @staticmethod
    def read(path):
        """The points (n, 3) of the file, without points that are not finite."""
        with open(path, "rb") as f:
            header = {}
            while True:
                key, *values = f.readline().decode().split()
                header[key] = values
                if key == "DATA":
                    break
            if header["DATA"] != ["binary"] or header["FIELDS"][:3] != ["x", "y", "z"] or set(header["SIZE"]) != {"4"}:
                raise ValueError(f"{path}: unsupported PCD layout")
            cloud = np.frombuffer(f.read(), np.float32).reshape(int(header["POINTS"][0]), len(header["FIELDS"]))[:, :3]
        return cloud[np.isfinite(cloud).all(1)].astype(float)

    @staticmethod
    def write(path, points):
        """Write points (n, 3) so that `read` returns them."""
        points = np.asarray(points, np.float32)
        header = (f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
                  f"COUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n")
        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(points.tobytes())
