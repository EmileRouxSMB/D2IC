# Memory — mechanics

## Rôle du dossier

Faire le lien entre les déplacements nodaux DIC et les lois locales aux points
de Gauss, sans dépendre du traitement des images.

## Fichiers importants

- `quadrature.py` : règle 2x2 et dérivées Q4.
- `deformation_gradient.py` : calcul compatible de `F` depuis les fonctions de forme.
- `history.py` : vectorisation de l'intégration sur éléments et points de Gauss.
- `equilibrium.py` : forces internes Q4, résidu faible et résultantes de traction sur les faces.
- `equilibrium_constraints.py` : masque des équations dont les réactions de mors sont inconnues.

## Concepts physiques ou numériques

Le gradient est calculé par rapport au maillage de référence. Les quatre points
de Gauss Q4 sont intégrés indépendamment par `vmap`; le temps est traité par
`lax.scan` dans `d2ic.constitutive`.

## Conventions

`F_history` a la forme `(temps, éléments, gauss, 2, 2)`. Les nœuds Q4 doivent
être ordonnés autour de l'élément avec une jacobienne de référence positive.

## Points d’attention

La reconstruction LSQ nodale de `physics/strain.py` n'est pas utilisée pour la
mécanique. Un déterminant de jacobienne négatif signale une connectivité Q4
inversée et doit être détecté avant l'intégration.
