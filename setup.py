from setuptools import setup

setup(name='D2IC',
      version='0.1',
      description="A differentiable framework for full-field kinematic identification.",
      long_description="",
      author='Emile Roux',
      author_email='emile.roux@univ-smb.fr',
      license='GPL v3',
      packages=['D2IC'],
      zip_safe=False,
      install_requires=[
          "jax",
          "opencv-python",
          "numpy",
          "scipy",
          "matplotlib",
          "scikit-image",          
          "dm_pix", 
          "gmsh",
          "meshio",
          ],
      )
