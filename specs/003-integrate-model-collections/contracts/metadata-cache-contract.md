# Contract: Collection Metadata Cache

## Purpose

Define behavior for fast model and weight listing through cached collection metadata.

## Cache Contents

1. Cache schema version.
2. Refresh timestamp.
3. Backend package versions when available.
4. Approved collection identifiers.
5. Model identifiers per collection.
6. Weight identifiers and default-weight metadata per model.
7. Source and integrity metadata that is safe to expose.

## Listing Behavior

1. Normal listing reads local cached metadata.
2. Listing results include model identifier, collection source, available weight variants, and default-weight marker when known.
3. Listing must not import or inspect live backend catalogs during normal cached reads.
4. Missing unselected backends must not break cached listing for other collections.
5. Listing should complete within the spec performance target for typical local environments.

## Refresh Behavior

1. Refresh explicitly inspects installed supported backend catalogs.
2. Refresh rebuilds cached metadata for installed supported backends.
3. Refresh records backend versions and refresh timestamp.
4. Refresh may skip missing optional backends and record them as unavailable.
5. Refresh failures for one backend must be reported clearly; implementation may keep the last valid cache for unaffected collections.

## Release Validation Behavior

1. Release validation compares refreshed cache model counts against each installed supported backend's model listing.
2. Each installed supported backend must expose at least 90% of backend-listed models through the refreshed cache unless a documented waiver explains unsupported entries.
3. Count comparison evidence must be recorded with backend version and refresh timestamp.

## Failure Behavior

- Missing cache returns a clear diagnostic and suggests refresh.
- Corrupt cache fails safely and suggests refresh.
- Refresh with no installed selected backend reports missing backend guidance.
- Stale cache is allowed but should be visible through metadata timestamp/version fields.

## Compatibility

- Cache format changes must use `schema_version`.
- Cache records must not contain secrets, credentials, private dataset paths, or private user data.
