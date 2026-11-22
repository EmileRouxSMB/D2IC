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

## Cinématique et historique isotropes optimisés

`isotropic_history.py` fournit :
- une polaire 2D analytique et log/exp symétriques réguliers aux valeurs propres répétées ;
- le chemin sous-incrémental R(alpha) exp(alpha log U) F_old, avec rotation
  relative principale dans [-pi,pi] (les tours non observés ne sont pas inférables) ;
- les incréments matériels et orientations préparés (T,S,Ne,Ng,3/2x2),
  indépendants des paramètres matériau ;
- `integrate_prepared` accepte le noyau commun Voce/Swift et fournit les états
  aux frames (stress4,p,log_thickness), y compris
  l'état initial, utilisables pour reprendre un suffixe à paramètres fixés ;
- les opérateurs Q4 de référence et l'assemblage réutilisant ces opérateurs.

`start` est un indice temporel dynamique : les frames précédentes sont sautées
sans recompiler selon leur nombre. Leurs champs de sortie sont des placeholders ;
le coût doit réutiliser les efforts du préfixe en cache. Le cache est invalidé
après changement matériau ou déplacement dans ce préfixe. La cinématique, y
compris ses orientations cumulées, reste calculée sur le chemin complet.
Le modèle reste corotationnel logarithmique, sans décomposition multiplicative.

`prepare_history` accepte une orientation initiale optionnelle pour calculer un incrément isolé dans la même base matérielle que l’histoire complète.
