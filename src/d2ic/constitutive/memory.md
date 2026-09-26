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

## Retour isotrope optimisé

`isotropic_plane_stress.py` traite J2 isotrope avec durcissement Voce ou Swift.
Voce utilise `sigma0 + K*(1-exp(-p/eps0))`; Swift utilise la forme standard
`K*(eps0+p)**n`. Le retour radial et sigma33=0 sont résolus ensemble en deux inconnues
(Delta p, Delta e33), avec jacobienne analytique 2x2. Les contraintes sont
[11,22,33,12], les déformations imposées [e11,e22,gamma12]. Le résidu est
adimensionné par E ; Newton vise 1e-13 et le contrôle final accepte 1e-12.
Un while_loop batched s'arrête lorsque tous les points ont convergé (30 tours
maximum) ; une recherche de pas préserve Delta p >= 0. Les dérivées JVP/VJP
proviennent de l'équation implicite au point convergé, pas des itérations.
Le changement de branche élastique/plastique est non différentiable au seuil ;
les tests de sensibilité évitent exactement ce seuil. Le solveur Hill48 général
reste disponible et sert de référence pour les cas isotropes.
