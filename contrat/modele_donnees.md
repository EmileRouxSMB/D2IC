# Contrat de modèle de données — D2IC

## Entrées en mémoire

- Images de gris : référence et déformée (H, W), indexées (y, x). Masque ROI booléen (H, W), aligné sur l'image utilisée.
- Mesh : nodes_xy (Nn, 2) en (x, y) et elements (Ne, 4), indices de nœuds Q4. L'intégration demande un Jacobien de référence positif.
- MeshAssets : mesh, topologie indépendante des images, masque et PixelSampling optionnels. PixelSampling porte pixel_coords_ref (Np, 2), pixel_nodes (Np, 4), pixel_shapeN (Np, 4), tables de voisins et, sur demande du solveur local à ressorts, de rassemblement nœud–pixel.
- MeshDICConfig règle une image ; BatchConfig règle la séquence et les exports ; SequenceConfig, lu depuis TOML, ajoute chemins, maillage, binning et plateforme JAX. Ces configurations sont des dataclass immuables.
- Hill48Material porte élasticité, coefficients Hill et écrouissage. MaterialState porte contrainte, déformation plastique, déformation plastique équivalente, orientation et déformation logarithmique d'épaisseur ; ce sont des PyTrees JAX.

## Sorties en mémoire

- DICResult : points_xy (Np, 2) optionnel, displacement (Np, 2), strain (Np, 3) optionnel, diagnostics, history, fields nodaux et pixel_maps libres. MeshDICResult ajoute le maillage ; u_nodal est un alias de displacement.
- En FE-DIC, Np = Nn ; strain suit [E11, E22, E12], cisaillement tensoriel. Si sa reconstruction échoue ou manque de voisins, la méthode retourne actuellement des zéros et renseigne diagnostics.info.
- BatchResult.results est la liste ordonnée des résultats conservés, vide si keep_results=False. Les diagnostics sont des dictionnaires libres ; n_iters du CG ne code pas un statut de convergence.
- Mécanique : contraintes Voigt [s11,s22,s33,s12,s23,s13] ; déformations Voigt [e11,e22,e33,gamma12,gamma23,gamma13], cisaillements ingénieur, distincts de E12 DIC.

## Sorties sur disque

- NPZ par image : u_nodal (Nn, 2), strain (Nn, 3), et éventuellement discrepancy_ref (H, W).
- NPZ global si save_npz=True, indépendamment de keep_results : nodes_xy, elements, u_nodal (Nt, Nn, 2), strain (Nt, Nn, 3), masque, dimensions, binning, convention de cisaillement, chemins et paramètres ; discrepancy_ref (Nt, H, W) est optionnel. L'image de référence n'occupe pas une ligne de u_nodal.
- load_dic_history_npz lit nodes_xy, elements et u_nodal ou displacement ; il vérifie les dimensions Q4, convertit en float64/int32 et ajoute par défaut une référence à déplacement nul. Il ne valide pas tous les champs du NPZ global.
- VTU porte géométrie Q4, déplacement et déformation nodale ; PVD porte l'ordre et les temps des images. PNG est une sortie de visualisation.
- Le format NPZ global de compatibilité reste sans numéro de schéma; le magasin versionné porte sa version dans `metadata.json`. Les champs libres et diagnostics n'ont pas de validation générale à la création du résultat.

## Magasin versionné des résultats séquentiels

- `d2ic.io.result_store.ResultStore` persiste le maillage dans `mesh.npz`, le masque dans `roi.npz`, les champs nodaux dans `frames/frame_NNNN.npz`, les métadonnées de schéma dans `metadata.json`, la progression dans `progress.json` et la reprise dans `checkpoint.npz`.
- Schéma version 1 : `nodes_xy (Nn,2)`, `elements (Ne,3|4)`, `u_nodal (Nn,2)`, `strain (Nn,3)` avec cisaillement tensoriel E12. L'image de référence reste séparée.
- `open_result_store(path)` ouvre un magasin versionné ou un NPZ historique. `read_frame`, `read_history` et `read_node_history` donnent accès aux champs; cette dernière méthode ne matérialise que les valeurs du nœud sélectionné par frame.
- Le checkpoint porte le fingerprint des sources et réglages ainsi que les deux derniers déplacements nécessaires à la propagation. Le fingerprint inclut les chemins et le SHA-256 du contenu des images source (ainsi que la configuration), donc une image remplacée à taille constante invalide la reprise. `export_npz(path)` produit le NPZ monolithique de compatibilité via des tableaux temporaires mappés sur disque. Le checkpoint est écrit après les exports VTU/VTI/PNG de la frame et avant la progression afin qu’une frame orpheline ne soit jamais présentée comme validée. Avant reprise, le maillage régénéré doit conserver exactement la connectivité et ses coordonnées doivent rester à 1e-10 près.
- `progress.json` contient `status`, `completed_frames`, `total_frames`, `last_frame` et `updated_at`. Le temps vient de la provenance fournie par l'appelant, et `frame_times` est conservé dans les métadonnées quand il existe.

## Code de référence

- `src/d2ic/io/result_store.py` complète `src/d2ic/mesh/assets.py`, `src/d2ic/core/dataclasses.py`, `src/d2ic/io/config.py`, `src/d2ic/io/mechanical_npz.py` et `src/d2ic/pipelines/sequence_platehole.py`.
