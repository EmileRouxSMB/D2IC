# Memory — core

## Rôle du dossier

Définitions communes et contrats minimaux partagés par toutes les méthodes DIC.

## Fichiers importants

- `types.py` : alias de type pour tableaux JAX/NumPy.
- `dataclasses.py` : configurations, résultats DIC, diagnostics et configurations batch.
- `dic_base.py` : interface abstraite `prepare/run` commune aux pipelines DIC.
- `__init__.py` : API publique du noyau commun.

## Concepts physiques ou numériques

Les résultats DIC sont représentés par `points_xy`, `displacement`, `strain`, `fields` et `pixel_maps`, indépendamment du support scientifique utilisé (maillage, points, autre représentation future).

## Conventions

Les déplacements sont en coordonnées image `(ux, uy)` et les points en `(x, y)`. Les dataclasses restent légères pour être faciles à transporter dans les pipelines JAX.

## Points d’attention

Ne pas introduire ici de dépendance vers les solveurs, les maillages ou la visualisation afin d’éviter les imports circulaires.
