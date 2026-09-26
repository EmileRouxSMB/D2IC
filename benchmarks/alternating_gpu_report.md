# Optimisation de l’identification alternée — mesures du 25 septembre 2026

Les six axes sont implémentés dans `IdentVirual/alternating_identification.py`,
`constitutive/isotropic_plane_stress.py` et `mechanics/isotropic_history.py`.
Le solveur Hill48 général reste disponible comme référence indépendante.

## Protocole

RTX 2000 Ada Generation Laptop GPU (8 Gio), JAX/jaxlib 0.10.2, Python 3.12.12.
Cas réel : 19 images mesurées, 1 468 éléments Q4, 4 points de Gauss,
2 sous-incréments, 18 modes de correction, mécanique float64.
Les temps des kernels sont des médianes de trois exécutions synchronisées,
après compilation et échauffement. Les deux lois comparées suivent le même
chemin objectif. Les gains ci-dessous sont isolés : ils ne se multiplient pas.

## Résultats par axe

| Axe | Référence | Version optimisée | Résultat |
|---|---:|---:|---|
| 1. Sous-incréments objectifs | Interpolation linéaire de F | R(α) exp(α log U) | Rotation rigide sans plasticité parasite ; correction physique |
| 2. Newton isotrope spécialisé | Hill48 général : 11 212 ms | J2/Voce : 83,7 ms | **134×** |
| 3. Cinématique précalculée | 83,7 ms | 57,4 ms | **1,46×** |
| 3. Préfixe historique réutilisé | Gradient complet : 176,4 ms | Suffixe dès l’image 10 : 151,4 ms | **1,17×** |
| 4. Coûts et contrôles GPU | Historiques transférés pour calcul hôte | Réductions GPU, scalaires ou 18 composantes transférés | Transferts réduits ; pas de facteur isolé mesuré |
| 5. Gradient des 18 modes | Différences finies sur le nouveau kernel : 2 695 ms | Dérivée implicite : 176,4 ms | **15,3×** |
| 5. Sensibilités des 3 paramètres | Différences finies : 234,6 ms | JVP implicite : 57,2 ms | **4,10×** |
| 6. Photométrie mixte | Float64 : 1,425 ms | Mixte : 1,388 ms | **1,03×**, gain temporel marginal ; mémoire image divisée par deux |
| 6. Image résidente GPU | Copie depuis hôte : 1,963 ms | Image déjà sur GPU : 1,388 ms | **1,41×** sur cet appel isolé |

Le Newton résout deux inconnues (incrément plastique et déformation transverse),
avec Jacobienne analytique 2×2, amortissement et arrêt sur résidu. La dérivée
implicite évite de différencier les itérations. Le logarithme polaire 2D utilise
une formule régulière aux valeurs propres répétées, notamment à l’identité.
La spécialisation concerne J2 isotrope/Voce ; elle ne remplace pas une loi Hill48
anisotrope. Le cache est reconstruit après changement des paramètres et actualisé
après chaque correction acceptée. Les opérateurs géométriques Q4 sont fixes.

Les lots de 1, 2, 4 et 8 historiques coûtent respectivement 83,67, 82,59, 65,96 et
61,73 ms **par historique**. Le lot de 8 améliore le débit de 1,36×, mais demande
plus de mémoire et n’est pas nécessaire pour le gradient implicite unique.
Le défaut retenu conserve les images float32 sur GPU (environ 167 Mio contre
334 Mio en float64), avec coordonnées, mécanique et réductions en float64.
La photométrie isolée utilise une image synthétique 1920×1200 et 65 000 points.

## Fidélité scientifique et coût mémoire

- Écart maximal de contrainte au solveur général : **4,43×10⁻⁸ MPa**.
- Erreur relative des sensibilités matériau contre différences finies : **2,48×10⁻¹⁰**.
- Erreur relative du gradient des modes : **1,30×10⁻⁴** pour un pas de différence finie de 10⁻³ ; les tests utilisent aussi des pas plus petits.
- Écart maximal entre gradients complet et avec préfixe : **3,47×10⁻¹⁸**.
- Les rotations rigides et la dérivée à l’identité sont testées, ainsi que chargement, décharge et conservation de la mémoire plastique.

Le benchmark des modes emploie ½ moyenne((force nodale / 100 N)²) ; celui des
paramètres différencie les composantes de contrainte. Ces mesures n’isolent pas
chaque terme de l’objectif complet. La différentiation implicite consomme plus
de buffers : environ **311 Mio** temporaires contre **108 Mio** pour le gradient
par différences finies groupées par deux. Ce sont des estimations XLA des buffers
compilés, pas une mesure de la VRAM maximale du processus. Le préfixe réduit le
travail, mais pas sensiblement ces buffers dans cette implémentation.

## Validation de la boucle complète

Le cas réel termine les trois alternances configurées en **45,06 s**, préparation,
compilation et exports compris (hors import des modules). Les 57 corrections sont
acceptées. Le coût passe de **0,00338132769** à **0,00337042935**, avec décroissance
aux étapes acceptées. Les ajustements matériau convergent ; les 19 images sont
conservées dans l’historique et les exports. Paramètres obtenus :
`[484.353882, 1042.838291, 0.243987177]`.

**L’alternance globale n’a pas atteint son critère d’arrêt en trois passages.**
Ce temps ne constitue donc pas une mesure du temps jusqu’à convergence globale.
Aucune exécution complète équivalente de l’ancien programme n’a été chronométrée :
on ne donne pas de facteur d’accélération global.

Une seconde exécution avec lecture, binning, normalisation et interpolation des
images entièrement float64 prend 48,29 s. Par rapport au mode mixte, l’écart maximal
de déplacement est **2,99×10⁻¹³ pixel**, l’écart relatif maximal des paramètres
**2,94×10⁻¹¹**, et l’écart maximal de contrainte **1,77×10⁻⁸ MPa**. Tous les points
constitutifs convergent. Il s’agit d’une validation sur ce cas, pas d’une garantie
pour toute texture ou tout chargement. Ces deux temps complets sont des mesures
uniques, non une étude statistique du gain de précision mixte.

Les tests CPU couvrant histoire, alternance, précision, identification,
Hill48, mécanique, équilibre et exports : **30 réussis**.
La validation ciblée sur CUDA (histoire isotrope et boucle complète) :
**13 réussis**, après les dernières modifications. Trois avertissements de
dépréciation proviennent de tifffile avec NumPy 2.5.

## Reproduction et artefacts

Depuis la racine du projet :

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python D2IC/benchmarks/benchmark_alternating_gpu.py --real-data IdentVirual/outputs/virt_test/npz/fields_sequence.npz --repeats 3 --output D2IC/benchmarks/alternating_gpu_real_results.json
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python IdentVirual/alternating_identification.py --output /tmp/alternating_validation
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python IdentVirual/alternating_identification.py --output /tmp/alternating_float64_validation --photometric-precision float64
```

- [Mesures détaillées des kernels](alternating_gpu_real_results.json)
- [Validation complète](alternating_workflow_validation.json)
- [Comparaison des précisions](alternating_precision_validation.json)
- [Benchmark reproductible](benchmark_alternating_gpu.py)

Les temps dépendent du matériel, des dimensions et de l’activité concurrente.
La compilation est enregistrée séparément dans le JSON. Les historiques supposent
un état initial non déformé ; la rotation interpolée utilise l’angle principal
entre deux images et ne reconstruit pas des tours non observés.
