# Memory — benchmarks

## Rôle du dossier

Benchmarks reproductibles des performances numériques D²IC.

## Fichiers importants

- `benchmark_butterfly_cpu_vs_gpu.py` : comparaison CPU/GPU sur la séquence butterFly.

## Concepts physiques ou numériques

Le benchmark sépare préparation/compilation, temps DIC par image et temps total afin de rendre visible le coût d’amortissement XLA.

## Conventions

Les résultats JSON et figures générées sont des artefacts locaux ignorés par Git. Les mesures synchronisent explicitement les calculs JAX avant chronométrage.

## Points d’attention

Toujours enregistrer versions JAX/jaxlib, backend, GPU, précision et dimensions du problème avec les temps.

## Alternance mécanique GPU

`benchmark_alternating_gpu.py` compare le solveur Hill48 général et le kernel
isotrope sur LE MÊME chemin objectif de sous-incréments. Il mesure séparément
compilation, exécution synchronisée, précalcul cinématique, gradient implicite
versus différences finies, préfixe réutilisé, taille des lots, précision photo
et résidence hôte/device. Les erreurs de contraintes et de dérivées sont sauvées.
`--real-data IdentVirual/outputs/virt_test/npz/fields_sequence.npz` permet de
mesurer les dimensions et déplacements réels ; `--repeats` règle la médiane.
Les tailles mémoire sont les estimations de buffers compilés XLA, pas la VRAM
maximale de tout le processus. Les gains isolés ne doivent pas être multipliés
pour annoncer un gain global. Le script est également un exemple minimal des
API numériques optimisées, avec cas synthétique reproductible par défaut.

Le rapport `alternating_gpu_report.md` documente les six optimisations, leurs
mesures sur le cas réel et les limites (buffers AD, gains non multiplicatifs,
absence de convergence globale après trois alternances). Les JSON
`alternating_gpu_real_results.json`, `alternating_workflow_validation.json` et
`alternating_precision_validation.json` conservent les résultats correspondants.

`audit_dic_2026_09_25.md` : revue approfondie des incohérences DIC/alternance.
`audit_dic_consistency.py` reproduit les défauts de support, interpolation,
régularisation et non-finis sur CPU/CUDA (observations, pas tests d’acceptation).
`audit_dic_preparation.py` compare le scan pixel-nœud au tri stable hôte, avec
vérification des tables et synchronisation des mesures. Les JSON `audit_dic_*`
conservent les observations ; aucune extrapolation au temps global.
