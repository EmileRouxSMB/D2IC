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
- `abaqus_comparison.py` : synchronisation temporelle, interpolation T3/Q4 et rapports nodaux DIC--Abaqus.
- `mechanical_npz.py` : transfert du maillage et de l’histoire DIC vers le post-traitement mécanique.
- `result_store.py` : magasin séquentiel versionné, lecteur des archives historiques et export NPZ global de compatibilité.
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
Quand la comparaison Abaqus est activée, ces temps sont lus dans les noms
`*_t_<temps>.*`. `Abaqus_Displacement` est exprimé en pixels dans le repère
image, comme `Displacement`; l'erreur est toujours `DIC - Abaqus`.
La force interne assemblée aux nœuds est exportée uniquement comme le vecteur
`Nodal_internal_force`. La contrainte de Cauchy est exportée uniquement comme le
tenseur `Cauchy_stress`; ses composantes et les normes vectorielles sont laissées
aux filtres de ParaView. `Equivalent_stress` contient la contrainte équivalente
de von Mises calculée aux points de Gauss puis moyennée par élément.

Le binning image moyenne les blocs complets et tronque le bord incomplet. Le
binning de masque suit la règle FE conservative : un bloc est actif si au moins
un pixel source est actif; les blocs de bord incomplets sont conservés.

## Points d’attention

Le magasin de résultats utilise `metadata.json`, `mesh.npz`, `roi.npz`, `frames/frame_NNNN.npz`, `progress.json` et `checkpoint.npz`. Le maillage existant est préservé et doit correspondre à la topologie régénérée (indices exacts, coordonnées à 1e-10 près) avant reprise. Les champs nodaux sont `u_nodal (Nn,2)` et `strain (Nn,3)` avec E12 tensoriel. Le NPZ global reste un export construit à la demande.

Les masques TIFF à deux canaux sont lus comme niveau/index + alpha : le canal alpha est ignoré.

La configuration du backend JAX doit être faite avant les gros imports/compilations. Les fonctions I/O restent hors des kernels JAX.
Les métriques Abaqus excluent les nœuds non référencés par la connectivité DIC
et ceux dont l'interpolation est invalide. Une extrapolation n'est admise que si
la distance au bord Abaqus reste inférieure à la limite physique configurée.

Le masque de la séquence peut être désigné par `paths.mask_path` (absolu ou relatif au TOML), prioritaire sur `paths.mask_filename` (relatif au dossier des images).

La configuration TOML utilise par défaut keep_results=false hors comparaison Abaqus. `save_npz=true` active le magasin par image ; `export_legacy_npz=true` (défaut D2IC) crée en plus un NPZ global. Le GUI choisit `export_legacy_npz=false` et offre cet export à la demande.

`paths.image_paths` est une liste optionnelle de chemins d’images déformées, résolus contre le TOML. Dans ce mode, le pipeline utilise exactement cette liste au lieu du motif et du pas ; elle sert aux runs GUI figés.
