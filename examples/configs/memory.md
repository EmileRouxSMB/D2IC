# Memory — configurations d’exemples

## Rôle du dossier

Configurations TOML reproductibles pour les pipelines séquentiels.

## Fichiers importants

- `sequential_dic.toml` : paramètres de maillage, solveur, propagation et sorties.

## Concepts physiques ou numériques

Les paramètres distinguent la résolution géométrique du maillage, la
régularisation et les critères de convergence.

## Conventions

Les tailles sont exprimées en pixels sauf indication contraire.
Les chemins relatifs sont résolus depuis le dossier `configs/`. La
configuration PlateHole écrit dans `../outputs/sequential_dic_platehole/`.

## Points d’attention

Une modification de clé doit rester synchronisée avec `SequenceConfig`.
