# Memory — batch

## Rôle du dossier

Traitement séquentiel de séries d’images et gestion des sorties par frame.

## Fichiers importants

- `base.py` : squelette `before/sequence/end` pour un traitement batch.
- `mesh_based.py` : batch FE-DIC avec warm-start, propagation, exports NPZ/PNG et série VTU/PVD.
- `__init__.py` : exports batch.

## Concepts physiques ou numériques

Le batch garde l’image de référence et les assets constants, puis corrèle chaque image déformée. Les champs précédents peuvent initialiser la frame suivante.

## Conventions

Les résultats par frame sont des `DICResult`. Les exports hôte (`npz`, figures, VTU/PVD) restent en dehors des fonctions jittées. Le VTU porte les déplacements et déformations aux nœuds; le PVD ordonne les frames par indice temporel.

## Points d’attention

Éviter les synchronisations device→host sauf pour l’I/O explicite. Les imports de visualisation sont retardés au moment de l’export PNG.
