# Memory — D2IC

## Rôle du dossier

Projet de corrélation d’images numériques 2D différentiable avec JAX, organisé par concepts scientifiques.

## Fichiers importants

- `src/d2ic/mesh/` : géométrie, topologie, échantillonnage pixel et génération des maillages.
- `src/d2ic/numerics/` : solveurs JAX ZNCC, gradient conjugué et Gauss–Newton local.
- `src/d2ic/physics/` : résidus photométriques et reconstruction des déformations.
- `src/d2ic/pipelines/` : orchestration des séquences DIC.
- `examples/` : cas scientifiques exécutables et configurations.
- `benchmarks/` : mesures reproductibles CPU/GPU.
- `contrat/` : contrats courts du calcul, du design logiciel et des données.
- `tests/` : tests unitaires et validations scientifiques.

## Concepts physiques ou numériques

Le déplacement est interpolé sur un maillage quadrilatéral. L’identification minimise un résidu de conservation des niveaux de gris avec une régularisation nodale optionnelle. Les déformations de Green–Lagrange sont reconstruites par moindres carrés locaux.

## Conventions

Les coordonnées sont ordonnées `(x, y)` et les images indexées `(y, x)`. Le déplacement nodal a la forme `(Nn, 2)`. Les sorties de déformation utilisent actuellement la convention tensorielle `(E11, E22, E12)`.

## Points d’attention

Les modules plats historiques (`mesh_assets.py`, `solver_global_cg.py`, etc.) sont des alias de compatibilité. Toute nouvelle implémentation doit être ajoutée dans `mesh/`, `numerics/` ou `physics/`. La pile JAX/CUDA doit être versionnée de manière cohérente.
