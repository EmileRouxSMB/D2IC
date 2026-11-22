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

Maillage 2D quadrilatéral, coordonnées `(x, y)`, connectivité `(Ne, 4)`, pixels aux centres `(j + 0.5, i + 0.5)`. Après sélection des Q4, les nœuds sont compactés : chacun est référencé par au moins un élément et la connectivité est renumérotée.

## Points d’attention

Gmsh peut fournir des nœuds de contour qui ne sont utilisés par aucun Q4. La compaction se fait avant `make_mesh_assets`, donc avant les tables de voisinage, le solveur et les exports NPZ/VTK.
Chaque segment du polygone simplifié est maintenant une ligne CAD Gmsh distincte.
Cela contraint ses sommets pendant le maillage et donne des rangées de coupe
complètes pour le masque Meuwissen pleine largeur. La recombinaison Q4 peut
encore omettre un coin dans d'autres géométries : contrôler la largeur des
rangées avant de calculer une réaction.

L’association pixel–élément interroge un arbre spatial de pixels et conserve le premier élément contenant chaque pixel. Le gather dense nœud–pixel est optionnel : seul le solveur local à ressorts le demande. Son assemblage utilise un tri stable NumPy ; les tableaux finaux restent transférables comme PyTrees.

Gmsh est initialisé avec `interruptible=False` pour que le maillage fonctionne aussi dans le thread de service du GUI FastAPI, où l’installation de gestionnaires de signaux est interdite.

`mask_to_mesh_gmsh(..., numpy_output=True)` garde les nœuds et éléments sur CPU pour l’aperçu du GUI, qui ne construit pas les actifs JAX du solveur. La valeur par défaut conserve la sortie JAX pour le calcul DIC.
