# Memory — methods

## Rôle du dossier

Pipelines DIC de haut niveau qui préparent un domaine scientifique puis appellent les solveurs numériques.

## Fichiers importants

- `mesh_based.py` : DIC globale sur maillage quadrilatéral FE-DIC.
- `init_motion.py` : initialisation grossière par corrélation ZNCC aux centres d’éléments.
- `__init__.py` : exports des méthodes DIC.

## Concepts physiques ou numériques

Une méthode DIC associe une image de référence à un domaine fixe puis résout chaque image déformée. Le pipeline FE-DIC expose ses résultats via `DICResult`; les classes génériques permettent d’ajouter ultérieurement un autre support de champ avec la même interface.

## Conventions

Les méthodes gardent l’orchestration hors des kernels JAX. Les tableaux lourds sont préparés une fois dans `prepare`, puis réutilisés dans `run`.

## Points d’attention

Les anciens imports plats (`d2ic.dic_mesh_based`, `d2ic.dic_init_motion`) sont maintenus par alias dans `d2ic.__init__`.
