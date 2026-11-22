# Memory — propagation

## Rôle du dossier

Stratégies de warm-start temporel pour les séquences DIC.

## Fichiers importants

- `base.py` : interface `DisplacementPropagatorBase`.
- `previous.py` : réutilisation du déplacement précédent.
- `constant_velocity.py` : extrapolation à vitesse constante `2*u_prev - u_prevprev`.

## Concepts physiques ou numériques

La propagation ne change pas l’objectif DIC; elle fournit seulement une condition initiale plus proche pour accélérer la convergence.

## Conventions

Les déplacements propagés ont la forme `(N, 2)` en pixels image.

## Points d’attention

Les propagateurs ne doivent pas modifier les buffers fournis; le batch copie les warm-starts lorsque les solveurs donnent les arguments à JAX.
