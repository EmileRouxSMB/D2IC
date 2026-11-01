# Contrat de conception logicielle — D2IC

- src/d2ic/ est le paquet public. Les sous-paquets thématiques contiennent les implémentations ; les anciens chemins plats sont des alias de compatibilité dans d2ic.__init__.
- mesh/ construit maillage Q4, topologie et tables de pixels ; numerics/ contient les solveurs JAX ; physics/ produit résidus et déformations ; mechanics/ et constitutive/ portent cinématique et lois matériau.
- DICMeshBased conserve l'état d'un problème à référence fixe : prepare(ref_image, assets), puis run(def_image). Le solveur injecté compile et résout ; la méthode calcule ensuite déformations et diagnostics.
- BatchMeshBased parcourt les images dans l'ordre, propage une estimation nodale et réalise les exports demandés. run_sequence_from_config assemble TOML, images, maillage, solveurs et batch.
- io/ est la frontière avec le disque : TOML, images, NPZ, VTU/PVD et CLI. visualization/ produit les figures. Les noyaux numériques ne réalisent pas d'exports.
- Les opérations répétées sur tableaux utilisent JAX : jit, vmap, lax et différentiation automatique. La préparation géométrique, le pilotage des séquences et les exports utilisent aussi Python/NumPy.
- Les assets géométriques sont réutilisés entre images. Les solveurs sont compilés à la préparation ; changer une forme de tableau ou un argument structurel peut provoquer une nouvelle compilation.
- Les classes portent configurations, données et étapes avec état. Les équations restent des fonctions. Les données JAX du maillage et du matériau sont enregistrées comme PyTrees.
- Le pipeline lit une image à la fois. Les exports restent dans la boucle par image. keep_results=True conserve explicitement tous les résultats ; par défaut, la configuration TOML utilise False et reconstruit le NPZ global depuis les fichiers par image.
- Toute modification d'équation doit conserver les conventions du contrat scientifique et vérifier au minimum un cas analytique, les dimensions et la compatibilité jit.
