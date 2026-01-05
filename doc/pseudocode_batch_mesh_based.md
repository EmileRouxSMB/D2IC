# Pseudo-code — `BatchMeshBased` (mesh-based DIC)

Ce pseudo-code décrit la logique de haut niveau de `BatchMeshBased` (module `D2IC/d2ic/batch_mesh_based.py`) pour une séquence d’images.

## Objets principaux

- `BatchMeshBased`
  - `ref_image`: image de référence
  - `assets`: `MeshAssets` (mesh + pré-calculs)
  - `dic_mesh`: pipeline DIC principal (`DICMeshBased`)
  - `dic_local` *(optionnel)*: raffinement local (`DICMeshBased`)
  - `propagator` *(optionnel)*: stratégie warm-start (`DisplacementPropagatorBase`)
  - `config`: `BatchConfig`

## Contrat global (`BatchBase.run`)

```text
function BatchBase.run(images):
    before(images)
    result = sequence(images)
    result = end(result)
    return result
```

## `BatchMeshBased.before(images)`

```text
function BatchMeshBased.before(images):
    dic_mesh.prepare(ref_image, assets)

    if dic_local is not None:
        dic_local.prepare(ref_image, assets)

    state.is_prepared = True
```

## `BatchMeshBased.sequence(images)`

```text
function BatchMeshBased.sequence(images):
    assert state.is_prepared

    # --- options sortie (npz par frame) ---
    if config.save_per_frame:
        per_frame_dir = config.per_frame_dir or CWD/_outputs/per_frame_fields
        mkdir(per_frame_dir)

    # --- options export PNG (plots) ---
    if config.export_png:
        png_dir = config.png_dir or CWD/_outputs/png
        mkdir(png_dir)
        plot_fields = config.plot_fields (+ "discrepancy" si demandé)
        plot params = (cmap, alpha, plot_mesh, dpi, binning, projection)
        pixel_assets = assets.pixel_data or build_pixel_assets(mesh, ref_image, binning)

    per_frame_results = []
    u_prev = None
    u_prevprev = None

    for each frame index k and image Idef in images:
        # --- logging ---
        if config.progress or config.verbose:
            print("[Batch] Frame k/n: start")

        # --- warm-start (initial guess) ---
        if propagator is not None:
            u_warm = propagator.propagate(u_prev, u_prevprev)
        else if config.warm_start_from_previous:
            u_warm = u_prev
        else:
            u_warm = None

        if u_warm is not None:
            dic_mesh.set_initial_guess(u_warm)

        # --- solve principal (CG / global) ---
        cg_res = dic_mesh.run(Idef)
        res = cg_res

        # --- raffinement local optionnel ---
        if dic_local is not None:
            dic_local.set_initial_guess(cg_res.u_nodal)
            res = dic_local.run(Idef)

        per_frame_results.append(res)

        # --- sauvegarde npz optionnelle ---
        if config.save_per_frame:
            save_npz(per_frame_dir/frame_k.npz, u_nodal=res.u_nodal, strain=res.strain)

        # --- mise à jour historique warm-start ---
        u_prevprev = u_prev
        u_prev = res.u_nodal

        if config.progress or config.verbose:
            print("[Batch] Frame k/n: done")

        # --- export PNG optionnel ---
        if config.export_png and (config.export_frames is None or k in config.export_frames):
            plotter = DICPlotter(result=res, mesh=assets.mesh, def_image=Idef, ref_image=ref_image, ...)
            for field in plot_fields:
                fig = plotter.plot(field, ...)
                save_png(png_dir/frame_k_field.png)
            close_fig(fig)

    diagnostics = {
        stage: "batch_mesh_based",
        n_frames: len(per_frame_results),
        warm_start_from_previous: config.warm_start_from_previous,
    }
    return BatchResult(results=per_frame_results, diagnostics=diagnostics)
```

## `BatchMeshBased.end(result)`

```text
function BatchMeshBased.end(result):
    # Placeholder: post-processing
    diagnostics = result.diagnostics + {"post": "end() placeholder"}
    return BatchResult(results=result.results, diagnostics=diagnostics)
```

