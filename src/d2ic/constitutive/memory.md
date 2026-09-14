# Memory — constitutive

## Rôle du dossier

Intégration locale JAX de lois métalliques anisotropes, indépendante de la DIC.

## Fichiers importants

- `return_mapping.py` : GRMA implicite 3D Hill48.
- `plane_stress.py` : élimination locale de la déformation hors plan.
- `corotational.py` : enveloppe grandes rotations et intégration temporelle.
- `state.py` : paramètres et variables d'état PyTree.
- `hill48.py`, `hardening.py` : surface de charge et écrouissage isotrope.

## Concepts physiques ou numériques

La loi locale est une plasticité associée Hill48 à écrouissage isotrope. Le
Newton résout simultanément l'incrément de contrainte et le multiplicateur
plastique. Les grandes rotations sont traitées par un incrément logarithmique
corotationnel; il ne s'agit pas d'une décomposition multiplicative `Fe @ Fp`.

## Conventions

Contraintes Voigt : `[s11,s22,s33,s12,s23,s13]`. Déformations Voigt :
`[e11,e22,e33,gamma12,gamma23,gamma13]`, avec cisaillements ingénieur. Les
contraintes et variables plastiques sont stockées dans les axes matière.

## Points d’attention

Activer `jax_enable_x64` pour l'identification. La surface Hill est normalisée
par `G+H`, comme le RMA historique. Toute nouvelle surface de charge doit être
homogène de degré un et fournir une contrainte équivalente scalaire.
