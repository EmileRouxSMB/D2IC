# Memory — tests

## Rôle du dossier

Tests scientifiques, compatibilité JAX et validations de séquences.

## Fichiers importants

- `test_correlation_roi.py` : érosion euclidienne, trous, bord image, unités avec
  binning, maillage inchangé, sélection des observations et export des deux ROI.

- `test_binning.py` : convention commune de binning des images et masques FE/GP.
- `test_mesh_data_model.py` : invariants de topologie et compatibilité PyTree.
- `test_strain_smoke.py` : extension homogène analytique.
- `test_isotropic_gpu_history.py` : équivalence J2/Hill isotrope, rotations, jacobienne analytique, sensibilités implicites, identité et suffixes corrigés.
- `test_constitutive_hill48.py` : retour implicite, contrainte plane, objectivité et différentiation.
- `test_constitutive_paraview.py` : champs élémentaires VTU et timesteps PVD constitutifs.
- `test_abaqus_comparison.py` : repères caméra, temps exacts, interpolation affine T3/Q4, extrapolation de bord et exclusion des nœuds isolés.
- `test_mechanics_constitutive.py` : gradient de transformation affine Q4 aux points de Gauss.
- `test_alternating_identification.py` : champs virtuels lisses et différentiabilité du résidu faible.
- `test_sequence_frame_limit.py` : ROI, stride et limite temporelle du workflow DP600L1 alterné.
- `test_sequence_mask_path.py` : masque externe relatif ou absolu, priorité et absence de désignation.
- `test_mechanical_equilibrium.py` : équilibre sous contrainte uniforme et traction analytique de face.
- `test_equilibrium_constraints.py` : exclusion contrôlée des réactions inconnues du résidu.
- `test_interpolation_boundaries.py` : interpolation rectangulaire, sorties de domaine et diagnostics device.
- Les tests importent les sous-packages canoniques; les alias historiques sont vérifiés séparément.
- `test_validation_imageserie_*.py` : validations de bout en bout.

- `test_preparation_streaming.py` : mapping spatial, gather local optionnel et assemblage NPZ progressif.

## Concepts physiques ou numériques

Les tests contrôlent les formes, les limites analytiques, la compilation et les statistiques d’erreur sur images de référence.

## Conventions

Les validations longues portent le marqueur pytest `slow`.

## Points d’attention

`conftest.py` rend accessibles les packages locaux `d2ic` et `im_jax`.

`test_global_cg_steps.py` vérifie une translation subpixel en 8/16 bits,
la décroissance du coût et le rejet du recouvrement nul.

`test_abaqus_single_element_constitutive.py` compares all 24 Job-1Elt
increments to `job_1elt_abaqus.json`: in-plane S, LE and PE, PEEQ, PEMAG,
yield activation, NFORC and RF in the ODB convention. Regenerate the JSON
with `tools/export_abaqus_single_element.py` from the INP and ODB in
`virtual-dic/data/SimuButterFly`.

- `test_elastic_regularization.py` : translation et rotation rigides finies
  d'énergie nulle, étirement uniaxial analytique, compatibilité JIT.
