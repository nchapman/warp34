/**
 * Subpath-aware URL resolution so the embedded dashboard works both at `/`
 * and behind reverse-proxy subpaths (e.g. llama-swap's `/upstream/<model>/`,
 * nginx, or Caddy).
 */

const SPA_ROUTE_SUFFIX = /\/(?:playground(?:\/.*)?|connect(?:\/.*)?)$/;

/**
 * Derive the mount prefix (no trailing slash, or `""` at root) from the
 * current browser pathname by stripping known SPA routes.
 */
export function detectBasePath(
  pathname: string = typeof window !== "undefined"
    ? window.location.pathname
    : "/",
): string {
  const withoutRoute = pathname.replace(SPA_ROUTE_SUFFIX, "");
  const trimmed = withoutRoute.replace(/\/+$/, "");
  return trimmed === "" || trimmed === "/" ? "" : trimmed;
}

/** Resolve an engine endpoint path (e.g. `/api/metrics`) against the mount prefix. */
export function apiUrl(
  path: string,
  basePath: string = detectBasePath(),
): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${basePath}${normalized}`;
}

/** Resolve the full origin + mount prefix (e.g. `http://host:8080/upstream/model`). */
export function baseOriginUrl(
  origin: string = typeof window !== "undefined" ? window.location.origin : "",
  basePath: string = detectBasePath(),
): string {
  return `${origin.replace(/\/+$/, "")}${basePath}`;
}
