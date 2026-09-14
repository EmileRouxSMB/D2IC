# Memory — package d2ic

## Rôle du dossier

API publique du package et compatibilité avec les anciens chemins d’import. Les implémentations scientifiques sont rangées dans des sous-packages thématiques.

## Fichiers importants

- `__init__.py` : API publique de haut niveau et alias de compatibilité vers les anciens chemins plats, sans fichiers doublons au niveau racine.
- `core/` : types, dataclasses et interfaces communes.
- `methods/` : pipelines DIC de haut niveau (`mesh_based`, `init_motion`).
- `batch/` : exécution séquentielle et exports par frame.
- `propagation/` : stratégies de warm-start temporel.
- `mesh/`, `numerics/`, `physics/`, `pipelines/` : implémentations géométriques, solveurs JAX, observables physiques et cas orchestrés.
- `visualization/` : tracés et projections des champs.
- `io/` : lecture image, configuration TOML et CLI.
- `constitutive/` : intégration locale JAX des métaux anisotropes.
- `mechanics/` : cinématique Q4 aux points de Gauss et couplage avec les histoires DIC.
- Anciens chemins plats (`d2ic.strain`, `d2ic.mask2mesh`, `d2ic.solver_*`, etc.) : fournis par alias de modules dans `__init__.py`.

## Concepts physiques ou numériques

`DICResult` est l’objet commun entre supports scientifiques. La FE-DIC utilise `MeshAssets.mesh` comme géométrie unique. Les classes génériques (`DICBase`, `DICResult`, diagnostics, champs et cartes pixel) restent disponibles pour ajouter une future méthode sans dépendre du maillage. Les solveurs numériques restent dans `numerics/`, les quantités mécaniques et photométriques dans `physics/`.

## Conventions

Les imports internes doivent viser les sous-packages canoniques, pas les alias plats. Les anciens chemins (`d2ic.dataclasses`, `d2ic.dic_mesh_based`, `d2ic.plotter`, `d2ic.strain`, `d2ic.mask2mesh`, etc.) sont fournis par `d2ic.__init__` pour les notebooks et scripts existants.

## Points d’attention

Lorsqu’un nouveau module important est ajouté, créer ou mettre à jour le `memory.md` du sous-package concerné. Éviter de remettre de nouvelles implémentations scientifiques au niveau racine.
