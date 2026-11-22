# Memory — transfert mécanique NPZ

`mechanical_npz.py` charge le maillage Q4 et l'histoire des déplacements depuis
les archives D2IC. Il ajoute par défaut l'état de référence `u=0`, absent de
`fields_sequence.npz`. Le NPZ reste un format de transfert, pas un état JAX.
