# Contrat numérique et scientifique — D2IC

## DIC 2D

- Repère image : coordonnées (x, y), tableaux indexés [y, x], centres des pixels à (colonne + 0,5, ligne + 0,5). Déplacements en pixels.
- Domaine : ROI maillée en quadrilatères Q4 ; l’affectation cherche les pixels proches de chaque élément et retient le premier élément contenant le pixel. Sur chaque pixel p affecté à un élément : u(p) = Σ N_a(p) u_a, avec quatre fonctions de forme bilinéaires.
- Résidu : r_p(u) = I_def(p + u(p)) - I_ref(p). Le solveur global minimise J_img = Σ(r_p²)/(2 N_valide) sur les pixels valides ; il renvoie +∞ si aucun pixel n'est valide.
- Validité : les points de référence et déformés doivent rester dans leurs images. Le support et N_valide varient avec u ; un pixel qui sort peut donc améliorer artificiellement le coût.
- Interpolation : cubique par défaut ; linéaire en option. Le chemin linéaire dm_pix et l'interpolateur interne n'appliquent actuellement pas le même décalage de centre de pixel.
- Objectif global : J(u) = J_img(u) + α J_reg(u), avec α = reg_strength.
  - elastic (défaut) : F = I + ∇_X u et E = (FᵀF - I)/2 aux quatre points de Gauss Q4. Énergie de membrane isotrope ½ λ tr(E)² + E:E ; module de cisaillement fixé à 1, coefficient de Poisson fixé à 0,3, poids d'aire normalisés et facteur d'échelle calculé depuis le graphe nodal.
  - spring : somme pondérée des ||u_i - u_j||² sur les voisins, avec facteur 1/4 sur les arêtes orientées stockées.
- Solveur global : gradient JAX, gradient conjugué non linéaire, redémarrage d'une direction non descendante et recherche d'Armijo limitée à 24 essais. Arrêt sur ||∇J|| < tol, refus du pas ou max_iters.
- Initialisation optionnelle : recherche de translation par ZNCC autour des centres d'éléments, puis projection aux nœuds.
- Raffinement local : avec spring, balayages Jacobi de Gauss–Newton amortis en nombre fixe, sans test de décroissance du coût global ; sa normalisation photométrique diffère du global. Avec elastic, le solveur local délègue actuellement à un nouveau CG global.
- Déformation nodale de sortie : ∇u estimé par moindres carrés sur les voisins, poids gaussiens si strain_gauge_length > 0 ; E = (FᵀF-I)/2, stocké [E11, E22, E12] avec E12 tensoriel. Cette reconstruction nodale ne sert pas à la mécanique.

## Mécanique et matériau

- Mécanique Q4 : F = I + ∇_X u est recalculé aux quatre points de Gauss depuis les déplacements nodaux ; historique (temps, éléments, 4, 2, 2).
- Hill48 : plasticité associée, σ_eq = sqrt(σᵀ P σ), matrice P issue de [F,G,H,L,M,N] et normalisée par G+H. Retour implicite par Newton sur l'incrément de contrainte et le multiplicateur plastique ; sortie avec tangente et indicateur de convergence.
- Écrouissage : Voce σ0 + K(1-exp(-p/ε0)), Swift/Nadai σ0 + K(ε0+p)^n, ou table [p, limite] interpolée linéairement.
- Contrainte plane : Newton ajuste l'incrément hors plan pour imposer σ33 = 0 ; la tangente en plan est condensée. L'histoire corotationnelle utilise lax.scan.
- Précision : la DIC suit la précision JAX configurée ; les validations constitutives et l'identification requièrent jax_enable_x64.

## Code de référence

- DIC : src/d2ic/numerics/, src/d2ic/physics/strain.py.
- Mécanique : src/d2ic/mechanics/, src/d2ic/constitutive/.
