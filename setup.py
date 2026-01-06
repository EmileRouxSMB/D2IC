from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf8")

setup(
    name="D2IC",
    version="0.1",
    description="A differentiable framework for full-field kinematic identification.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Emile Roux",
    author_email="emile.roux@univ-smb.fr",
    license="GPL v3",
    packages=find_packages(include=["d2ic*"]),
    zip_safe=False,
    install_requires=[
        "jax",
        "opencv-python",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-image",
        "dm_pix",
        "im-jax"
        "gmsh",
        "meshio",
    ],
)
