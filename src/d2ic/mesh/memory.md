# Memory — mesh

## Rôle du dossier

Implémentation canonique de la géométrie, des topologies, de l’échantillonnage pixel et de la génération des maillages ROI.

## Fichiers importants

- `assets.py` : `Mesh`, topologies, tables d’adjacence et PyTrees.
- `pixel_sampling.py` : association pixel–élément, fonctions de forme et gathers nodaux.
- `generation.py` : maillages quadrilatéraux structurés ou générés par Gmsh depuis un masque.
- `geometry.py`, `topology.py`, `sampling.py` : façades thématiques légères vers les objets précédents.
- `__init__.py` : API publique du sous-package.

## Concepts physiques ou numériques

La géométrie du maillage est indépendante des images. `PixelSampling` dépend de l’image de référence, de la ROI et du binning. La topologie de déformation et le graphe de régularisation sont deux objets distincts.

## Conventions

Maillage 2D quadrilatéral, coordonnées `(x, y)`, connectivité `(Ne, 4)`, pixels aux centres `(j + 0.5, i + 0.5)`.

## Points d’attention

La construction des assets mélange volontairement NumPy pour la préparation et JAX pour les kernels géométriques coûteux. Les tableaux finaux sont transférables comme PyTrees.
