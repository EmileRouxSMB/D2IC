# Memory — io

## Rôle du dossier

Lecture d’images, configuration TOML, sélection runtime JAX et interface CLI.

## Fichiers importants

- `decimation.py` : masque de sous-échantillonnage régulier sans filtrage ni modification de la grille image.

- `app_utils.py` : lecture image, masque ROI, configuration backend JAX.
- `binning.py` : binning commun des images et masques pour les méthodes FE et GP.
- `config.py` : parsing de la configuration séquentielle.
- `paraview.py` : export des maillages et champs cinématiques nodaux au format VTU/PVD.
- `constitutive_paraview.py` : champs constitutifs élémentaires et séquence PVD synchronisée.
- `mechanical_npz.py` : transfert du maillage et de l’histoire DIC vers le post-traitement mécanique.
- `cli.py` : point d’entrée `d2ic run-sequence`.
- `__init__.py` : API publique I/O.

## Concepts physiques ou numériques

Le prétraitement image fixe l’échelle photométrique et les masques utilisés par les méthodes DIC. La configuration décrit les paramètres expérimentaux et numériques d’une séquence.

## Conventions

Les images lues sont converties en niveaux de gris `float32`. Les chemins TOML relatifs sont résolus par rapport au fichier de configuration.

Les déformations VTU suivent l’ordre `[E11, E22, E12]`, avec `E12` tensoriel; les vecteurs 2D sont complétés par une composante z nulle.
Le déplacement est exporté comme un unique vecteur VTK 3D
`Displacement=[Ux, Uy, 0]`. Les résultats VTU et les images VTI utilisent deux
collections PVD distinctes partageant les mêmes temps. ParaView conserve ainsi
une colormap, une opacité et une représentation indépendantes pour chaque source.

Le binning image moyenne les blocs complets et tronque le bord incomplet. Le
binning de masque suit la règle FE conservative : un bloc est actif si au moins
un pixel source est actif; les blocs de bord incomplets sont conservés.

## Points d’attention

La configuration du backend JAX doit être faite avant les gros imports/compilations. Les fonctions I/O restent hors des kernels JAX.
