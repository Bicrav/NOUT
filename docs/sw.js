// Offline in a field, but never stale at home.
//
// The first version of this served the cache first, which is wrong for the page
// itself: once a phone had it, a newly published version could not get through —
// reported, and rightly so. The page is now fetched from the network when there is
// any, with the cache kept fresh behind it and used only when the network fails.
// Everything else (icon, manifest) is static, so cache-first suits it.
const CACHE = "nout-pointing-v4";
const FILES = ["./", "./index.html", "./icon.png", "./manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES))
              .then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const isPage = e.request.mode === "navigate"
                 || e.request.url.endsWith("/")
                 || e.request.url.endsWith("index.html");
  if (isPage) {
    e.respondWith(
      fetch(e.request).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put("./index.html", copy)).catch(() => {});
        return res;
      }).catch(() => caches.match("./index.html", { ignoreSearch: true }))
    );
    return;
  }
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) =>
      hit || fetch(e.request).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
    )
  );
});
