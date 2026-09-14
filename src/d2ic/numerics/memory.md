# Memory — numerics

## Rôle du dossier

Implémentation canonique des solveurs numériques JAX de D²IC.

## Fichiers importants

- `base.py` : protocoles d’état et contrat fonctionnel des solveurs.
- `translation_zncc.py` : initialisation de translation par corrélation ZNCC.
- `global_cg.py` : minimisation globale photométrique par gradient conjugué non linéaire.
- `local_gauss_newton.py` : raffinement nodal par balayages Gauss–Newton/Jacobi.
- `__init__.py` : API publique numérique.

## Concepts physiques ou numériques

Le solveur global minimise l’énergie photométrique normalisée par le nombre de pixels valides, plus une énergie de ressort nodale. Le solveur local assemble des mises à jour nodales depuis les pixels adjacents.

## Conventions

Les tailles de voisinage, de gather, le nombre maximal d’itérations et le mode d’interpolation sont structurels pour les kernels compilés. Les images sont transposées en `(x, y)` dans les solveurs.

## Points d’attention

Les boucles critiques utilisent `lax.while_loop` ou `lax.fori_loop`. Éviter les conversions NumPy et les synchronisations host/device dans `solve`.


Le warp photométrique préfiltre une seule fois les coefficients de la spline
bicubique. Son Jacobien est obtenu par différence centrée à `1e-3 px` sur cette
même spline, et non par interpolation d'une image `np.gradient`; cette
cohérence évite les fausses stagnations près du minimum. Les essais de recherche
linéaire ne calculent que l'intensité warpée. Les applications doivent éroder la
ROI d'au moins le rayon du support bicubique pour empêcher les interpolations de
traverser un bord de matière ou un trou du masque.

