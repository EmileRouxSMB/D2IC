# Memory — pipelines

## Rôle du dossier

Orchestration de cas DIC complets à partir de configurations et d’images.

## Fichiers importants

- `sequence_platehole.py` : orchestration canonique du pipeline séquentiel avec génération de maillage et exports.

## Concepts physiques ou numériques

Le pipeline construit obligatoirement le maillage avec Gmsh, prépare les solveurs
puis propage le déplacement initial entre images.
Le sous-échantillonnage est contrôlé par `paths.image_stride` et inclut la référence dans la sélection.
La table optionnelle `[abaqus_comparison]` enrichit les VTU après résolution et
écrit les erreurs nodales par image en JSON/CSV. Les PVD maillage et image sont
alors datés avec les temps exacts codés dans les noms d'images.

## Conventions

Image coordinates are `(column, row)`, with rows increasing downwards.
Matplotlib uses `origin="upper"` for both images and field overlays, matching
the TIFF orientation without transforming displacement or strain values.

Le maillage utilisé par les solveurs provient exclusivement de `MeshAssets.mesh`.

## Points d’attention

Les impressions et exports restent hors des kernels JAX. Une erreur Gmsh est
fatale : aucun mailleur de repli n’est utilisé.

La régularisation par défaut est `elastic` (Green–Lagrange isotrope) ;
`spring` conserve l'ancienne pénalité entre nœuds. Si un raffinement local
est demandé avec `elastic`, il poursuit l'objectif global par CG, car la
mise à jour Jacobi locale existante est spécifique aux ressorts.

Le pipeline lit le masque résolu dans `cfg.mask_path`, ce qui permet de conserver le masque hors du dossier des images visibles.

Le pipeline lit une image déformée à la fois. Si `save_npz=True`, chaque résultat est écrit dans un magasin versionné par image, avec progression et checkpoint contenant les deux derniers déplacements. Le fingerprint inclut les empreintes SHA-256 des images source; l’export global est remplacé atomiquement. Le NPZ global reste un export de compatibilité reconstruit depuis ce magasin.

Quand `paths.image_paths` existe, la séquence DIC suit ces chemins explicites et leur ordre. Le motif et le pas restent disponibles pour les configurations historiques sans liste explicite.
