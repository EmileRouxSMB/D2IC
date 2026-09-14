# Memory — pipelines

## Rôle du dossier

Orchestration de cas DIC complets à partir de configurations et d’images.

## Fichiers importants

- `sequence_platehole.py` : orchestration canonique du pipeline séquentiel avec génération de maillage et exports.

## Concepts physiques ou numériques

Le pipeline construit obligatoirement le maillage avec Gmsh, prépare les solveurs
puis propage le déplacement initial entre images.

## Conventions

Le maillage utilisé par les solveurs provient exclusivement de `MeshAssets.mesh`.

## Points d’attention

Les impressions et exports restent hors des kernels JAX. Une erreur Gmsh est
fatale : aucun mailleur de repli n’est utilisé.
