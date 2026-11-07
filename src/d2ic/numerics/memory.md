# Memory — numerics

## Rôle du dossier

Implémentation canonique des solveurs numériques JAX de D²IC.

## Fichiers importants

- `base.py` : protocoles d’état et contrat fonctionnel des solveurs.
- `translation_zncc.py` : initialisation de translation par corrélation ZNCC.
- `global_cg.py` : minimisation globale photométrique par gradient conjugué non linéaire.
- `local_gauss_newton.py` : raffinement nodal par balayages Gauss–Newton/Jacobi.
- `adaptive_mechanical_gp.py` : assemblage faible Q4 linéaire et kernels GP pour la variance spatiale du résidu.
- `mechanical_gp_solver.py` : identification conjointe déplacement/GP et variante mécanique uniforme.
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


Le CG borne chaque essai initial à 1 pixel par nœud, puis réduit le pas par
Armijo. Un échec de recherche conserve le dernier champ accepté ; une absence
totale de recouvrement a un coût infini. Ces choix évitent les sauts hors image
avec les intensités TIFF 16 bits.

## Régularisation élastique isotrope

`elastic_regularization.py` calcule les gradients Q4 à quatre points de Gauss puis
l'énergie de Saint-Venant–Kirchhoff en déformation de Green–Lagrange. Cette
énergie s'annule pour toute translation ou rotation rigide plane finie. Le
rapport de Lamé correspond à un coefficient de Poisson de 0,3, avec module
de cisaillement unitaire. `global_cg.py` étalonne son échelle sur l'ancienne
pénalité de ressorts pour un étirement axial uniforme ; `reg_strength` reste
un poids numérique à calibrer sur les résidus et les zones de contrôle.

`adaptive_mechanical_gp.py` prend un tenseur constitutif explicite (le choix
contrainte plane/déformation plane est donc porté par l'appelant). Les résidus
sont en force par unité d'épaisseur et `force_scale` doit avoir la même unité.
Les coordonnées et `length_scale` GP partagent une unité. Le latent `g=log(s²)`
est borné à `[-20,20]`; la NLL inclut sa normalisation `d*g/2`. La variance de
Laplace est conditionnelle au modèle inducteur et à l'approximation quadratique.
`mechanical_gp_solver.py` exclut tous les nœuds de bord lorsque les tractions et
réactions ne sont pas fournies. Sa carte `exp(-g)` indique la pertinence locale
du modèle constitutif choisi, et non une violation de l'équilibre physique.
