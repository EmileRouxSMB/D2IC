# Memory — physics

## Rôle du dossier

Implémentation canonique des observables photométriques et mécaniques.

## Fichiers importants

- `discrepancy.py` : interpolation bilinéaire et carte `I_def(x + u) - I_ref(x)`.
- `strain.py` : gradient de déformation et déformation de Green–Lagrange par moindres carrés nodaux.
- `__init__.py` : API publique physique.

## Concepts physiques ou numériques

Le gradient de déplacement est identifié sur le voisinage nodal, éventuellement pondéré par une longueur de jauge gaussienne. La déformation finie vaut `E = 0.5 * (F.T @ F - I)`.

## Conventions

Les tenseurs ont la forme `(Nn, 2, 2)`. `green_lagrange_to_voigt` renvoie actuellement la convention tensorielle `(E11, E22, E12)`, sans facteur 2 sur le cisaillement.

## Points d’attention

`small_strain_nodes_lsq` réutilise actuellement Green–Lagrange et n’est équivalent à une petite déformation qu’à gradient faible. Les conventions de frontière des diagnostics photométriques doivent rester cohérentes avec celles des solveurs.
