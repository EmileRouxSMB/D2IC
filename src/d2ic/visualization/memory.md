# Memory — visualization

## Rôle du dossier

Visualisation des champs DIC, des déformations et des cartes photométriques.

## Fichiers importants

- `plotter.py` : `DICPlotter`, projection des champs sur image et rendu Matplotlib.
- `__init__.py` : export de `DICPlotter`.

## Concepts physiques ou numériques

Les champs peuvent être nodaux, définis sur points génériques ou projetés vers pixels selon le support disponible dans `DICResult.geometry`.

## Conventions

Les composantes standard sont `u1`, `u2`, `e11`, `e22`, `e12`; les champs utilisateur passent par `DICResult.fields`.

## Points d’attention

Ne pas importer Matplotlib depuis les kernels numériques. La visualisation peut convertir vers NumPy car elle est explicitement côté hôte.
