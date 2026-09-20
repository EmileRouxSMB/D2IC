# Memory — tests

## Rôle du dossier

Tests scientifiques, compatibilité JAX et validations de séquences.

## Fichiers importants

- `test_binning.py` : convention commune de binning des images et masques FE/GP.
- `test_mesh_data_model.py` : invariants de topologie et compatibilité PyTree.
- `test_strain_smoke.py` : extension homogène analytique.
- `test_constitutive_hill48.py` : retour implicite, contrainte plane, objectivité et différentiation.
- `test_constitutive_paraview.py` : champs élémentaires VTU et timesteps PVD constitutifs.
- `test_mechanics_constitutive.py` : gradient de transformation affine Q4 aux points de Gauss.
- `test_alternating_identification.py` : champs virtuels lisses et différentiabilité du résidu faible.
- `test_sequence_frame_limit.py` : ROI, stride et limite temporelle du workflow DP600L1 alterné.
- `test_mechanical_equilibrium.py` : équilibre sous contrainte uniforme et traction analytique de face.
- `test_equilibrium_constraints.py` : exclusion contrôlée des réactions inconnues du résidu.
- `test_interpolation_boundaries.py` : interpolation rectangulaire, sorties de domaine et diagnostics device.
- Les tests importent les sous-packages canoniques; les alias historiques sont vérifiés séparément.
- `test_validation_imageserie_*.py` : validations de bout en bout.

## Concepts physiques ou numériques

Les tests contrôlent les formes, les limites analytiques, la compilation et les statistiques d’erreur sur images de référence.

## Conventions

Les validations longues portent le marqueur pytest `slow`.

## Points d’attention

`conftest.py` rend accessibles les packages locaux `d2ic` et `im_jax`.
