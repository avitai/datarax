# Releasing Datarax

Datarax publishes through `.github/workflows/publish.yml`. No commit or tag push
creates a release by itself: publishing a GitHub Release runs the PyPI upload with
trusted publishing, and a manual `workflow_dispatch` run can target TestPyPI or
PyPI directly. Release timing and versioning stay under operator control.

## Release Checklist

1. Activate the local environment.

   ```bash
   source activate.sh
   ```

2. Bump the package version in `src/datarax/__init__.py`. The version is dynamic,
   so `uv.lock` does not change with the bump.
3. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date.
4. Run the release checks.

   ```bash
   uv lock --check
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check --strict dist/*
   ```

5. Commit the version and changelog updates, push, and read CI at the job level
   for that commit. Every workflow must be green before the tag exists.
6. Create and push an annotated tag from the exact release commit.

   ```bash
   target_sha=$(git rev-parse HEAD)
   git tag -a vX.Y.Z -m "datarax X.Y.Z"
   git push origin main vX.Y.Z
   ```

7. Create the GitHub Release from the tag with generated notes.

   ```bash
   gh release create vX.Y.Z --target "$target_sha" --generate-notes
   ```

   Publishing the release triggers the build and the PyPI upload.

8. Confirm the upload from a throwaway environment.

   ```bash
   uv venv /tmp/datarax-smoke && uv pip install --python /tmp/datarax-smoke datarax==X.Y.Z
   ```

## TestPyPI

Use the manual `workflow_dispatch` path in `publish.yml` with `target=testpypi`
when validating the trusted publishing setup before a real release. The same
path with `target=pypi` uploads to PyPI without a GitHub Release; prefer the
release path so the tag, the notes and the upload stay together.

## PyPI Trusted Publishing

PyPI must trust:

- Owner: `avitai`
- Repository: `datarax`
- Workflow: `publish.yml`
- Environment: `pypi`

If PyPI rejects the publish with `invalid-publisher`, verify the trusted
publisher registration before looking for repository secrets. The expected
publisher identity is:

```text
repo:avitai/datarax:environment:pypi
```

For TestPyPI, the expected environment is `testpypi`.
