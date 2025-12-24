Testing notes
=============

These smoke tests exercise the legacy D2IC motion initialization pipeline with
a synthetic image pair. Run the suite from the repository root:

```
pytest -q
```

The tests only rely on numpy (and the legacy modules themselves), so no extra
dependencies are required.
