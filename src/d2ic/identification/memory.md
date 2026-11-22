# Memory — identification

## Rôle du dossier

Coupler souplement la fidélité photométrique DIC et l'équilibre constitutif.

## Fichiers importants

- `alternating.py` : termes de coût différentiables, champs virtuels gaussiens et pas Adam.

## Concepts physiques ou numériques

L'identification alternée fixe successivement les déplacements puis les paramètres matériau. L'équilibre est projeté sur des champs virtuels lisses qui s'annulent aux mors.

## Conventions

Les déplacements sont nodaux `(Nn, 2)` en pixels. Les champs virtuels ont la forme `(Nv, Nn, 2)` et une norme nodale unité.

## Points d’attention

Ne jamais construire la jacobienne par rapport à tous les déplacements. Utiliser un gradient scalaire image par image et figer l'état constitutif entrant pour borner la mémoire.
