# Validation constitutive — Abaqus Job-1Elt

## Cas et méthode

Référence : `virtual-dic/data/SimuButterFly/Job-1Elt.inp` et `.odb`, générés
avec Abaqus/CAE 2022. Un élément CPS4R de 5 × 5 mm et d'épaisseur initiale
1 mm est chargé en traction jusqu'à 0,5 mm (24 incréments, 25 frames).
Le matériau a `E=210000 MPa`, `nu=0.3` et 94 points `*Plastic`
`[contrainte vraie, PEEQ]`. La PEEQ finale reste dans la table.

Les déplacements U des quatre noeuds donnent le gradient affine `F` à chaque
frame. `integrate_history` reçoit ces `F`, une loi Hill48 isotrope J2
`[F,G,H,L,M,N]=[0.5,0.5,0.5,1.5,1.5,1.5]` et exactement la table d'écrouissage
de l'INP, interpolée linéairement. JAX est exécuté en float64 sur CPU.
Les sorties ODB sont stockées dans `tests/job_1elt_abaqus.json`; le script
`tools/export_abaqus_single_element.py` les régénère avec `abaqus python`.

## Écarts maximaux sur les 24 frames chargées

| Champ | Écart absolu maximal | Valeur finale D2IC / Abaqus |
| --- | ---: | ---: |
| S22 | 0,000661 MPa | 698,0145 / 698,0138 MPa |
| S11 | 0,000937 MPa | 0,000961 / 0,000023 MPa |
| S12, S33 | 1,7e-12 / 0 MPa | voisins de zéro |
| PE11 | 2,40e-8 | -0,04599221 / -0,04599218 |
| PE22 | 1,66e-7 | 0,09198630 / 0,09198614 |
| PE12 | 3,1e-17 | voisins de zéro |
| PEEQ | 1,61e-7 | 0,09198630 / 0,09198614 |
| PEMAG | 1,66e-7 | 0,09198630 / 0,09198614 |
| LE11, LE22 | 3,2e-9, 2,7e-9 | -0,04698937, 0,09531018 |
| NFORC2 par noeud, convention ODB | 0,00147 N | 1586,396 / 1586,395 N |
| RF2 au point de pilotage | 0,00294 N | 3172,793 / 3172,790 N |

Les 24 retours locaux convergent (résidu maximal 2,31e-13). L'activation
`AC YIELD` coïncide sur toutes les frames ; elle débute à `t=0,02`.
Le test vérifie également les composantes NFORC1/2 des quatre noeuds,
les réactions RF2 des deux noeuds bas, LE12 et l'incompressibilité plastique.
Les écarts de LE sont des contrôles cinématiques puisque U est l'entrée de la
loi ; PEEQ et PEMAG coïncident ici en traction plastique monotone.

## Convention des efforts et limites

Dans cet ODB, `NFORC2 = 2,5 S22 / F22` à l'arrondi près à chaque frame.
La reconstruction des NFORC et de RF ci-dessus suit cette relation observée,
soit une intégrale de `sigma @ F^-T` sans facteur de volume courant. La routine
`forces_from_fields` de D2IC inclut ce facteur ; son résultat diffère de
2,112 N au maximum (0,133 % en fin de charge). Ce cas valide donc la loi
constitutive, mais ne tranche pas la convention d'assemblage des forces pour
un modèle mécanique plus général. La formule ODB est une inférence à partir
des champs exportés, pas une affirmation sur l'implémentation interne Abaqus.

Les composantes LE33 et PE33 exportées valent zéro pour cet élément 2D et ne
représentent pas l'évolution hors plan. D2IC prédit en fin de charge
`log(thickness ratio)=-0,04699126` et `PE33=-0,04599409`, qui ne peuvent donc
pas être confrontés directement à ces deux colonnes ODB. U et les sorties
liées au couplage du point de pilotage sont des données cinématiques ou de
liaison, et non des prédictions de l'intégration constitutive. Aucun essai
GPU n'a été effectué sur cette machine (backend JAX CPU uniquement).

Commande de reproduction depuis `D2IC` :

```text
abaqus python tools/export_abaqus_single_element.py ../virtual-dic/data/SimuButterFly/Job-1Elt.odb ../virtual-dic/data/SimuButterFly/Job-1Elt.inp tests/job_1elt_abaqus.json
python -m pytest tests/test_abaqus_single_element_constitutive.py -q
python examples/validate_abaqus_single_element.py
```
