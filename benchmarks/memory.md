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
