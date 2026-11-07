# Memory — examples

## Rôle du dossier

Cas scientifiques reproductibles et configurations directement exécutables.

## Fichiers importants

- `meuvissen_adaptive_mechanical_gp.py` : comparaison DIC libre, mécanique uniforme et GP adaptatif sur la séquence butterfly synthétique avec vérité Abaqus.
- `plot_meuvissen_gp_validation.py` : génération des cartes PNG de déplacement, erreur, confiance GP et incertitude à partir du NPZ de validation.

- `sequential_dic_platehole.py` : traitement séquentiel FE-DIC du cas PlateHole depuis TOML.
- `validation_sample3_translation.py` : validation de translation rigide sur Sample3.
- `constitutive_postprocess_platehole.py` : post-traitement total Hill48/Voce depuis le NPZ DIC.
- `configs/sequential_dic.toml` : paramètres du pipeline PlateHole.

## Concepts physiques ou numériques

Les exemples couvrent la génération de ROI, le maillage, l’identification séquentielle FE-DIC, le warm-start, la sortie ParaView constitutive synchronisée et la validation sur déplacement imposé.

## Conventions

Les chemins sont résolus relativement aux fichiers de configuration ou à la racine du projet. Les exemples écrivent leurs sorties à côté des scripts dans `outputs/<nom_exemple>/`, avec les sous-dossiers `png/`, `npz/` et `paraview/`.

## Points d’attention

Les exemples doivent importer l’API publique ou les sous-packages canoniques, jamais les alias plats de compatibilité.

`validate_abaqus_single_element.py` compares the complete Job-1Elt Abaqus
history to the D2IC local constitutive integrator and reports field errors.
