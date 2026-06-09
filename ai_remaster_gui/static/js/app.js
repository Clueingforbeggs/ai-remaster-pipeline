function draw(followLogs = false) {
  if (active === 'global') return drawGlobal(followLogs);
  if (active === 'settings') return drawSettings();
  if (active === 'cache') return drawCache();
  if (active === 'output') return drawOutput();
  if (active === 'shots') return drawShots(followLogs);
  if (active === 'references') return drawReferences(followLogs);
  if (active === 'colour') return drawColour(followLogs);
  if (active === 'recomp') return drawRecomp(followLogs);
  if (active === 'upscale') return drawUpscale(followLogs);

  return drawStage(stage(active), followLogs);
}

// Self-scheduling poll: wait for each refresh to finish before timing the next one.
// A plain setInterval would fire every 4s even when a state fetch is still in flight,
// stacking overlapping requests (and their ffprobe work) and saturating the server.
function scheduleNextRefresh() {
  setTimeout(async () => {
    try {
      await refresh();
    } finally {
      scheduleNextRefresh();
    }
  }, 4000);
}

refresh();
scheduleNextRefresh();
