/* mjorbit project page — interactions */

// Smooth-scroll past the hero when the indicator is clicked.
function scrollToContent() {
  const main = document.querySelector('.main-content');
  if (main) main.scrollIntoView({ behavior: 'smooth' });
}
window.scrollToContent = scrollToContent;

document.addEventListener('DOMContentLoaded', function () {
  /* ---- Reveal a video only once real media has loaded ----
     Until then the placeholder poster (frame background, or the banner for
     the hero) stays visible — so a missing .mp4 never shows a black box. */
  const hero = document.getElementById('bg-video');
  if (hero) {
    hero.addEventListener('loadeddata', () => hero.classList.add('loaded'));
    if (hero.readyState >= 2) hero.classList.add('loaded');
  }
  document.querySelectorAll('.example video').forEach((v) => {
    const reveal = () => {
      v.classList.add('loaded');
      const badge = v.closest('.video-frame')?.querySelector('.badge');
      if (badge) badge.style.display = 'none';
    };
    v.addEventListener('loadeddata', reveal);
    if (v.readyState >= 2) reveal();
  });

  /* ---- Click any example video to toggle fullscreen ---- */
  document.querySelectorAll('.example video, #bg-video').forEach((v) => {
    v.addEventListener('click', () => {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else if (v.requestFullscreen) {
        v.requestFullscreen();
      } else if (v.webkitEnterFullscreen) {
        v.webkitEnterFullscreen(); // iOS Safari
      }
    });
  });

  /* ---- Lazy-play example videos only while in view (saves bandwidth) ---- */
  const vids = document.querySelectorAll('.example video');
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const v = e.target;
          if (e.isIntersecting) {
            v.play().catch(() => {}); // no-op if no source yet
          } else if (!v.paused) {
            v.pause();
          }
        });
      },
      { threshold: 0.25 }
    );
    vids.forEach((v) => {
      v.removeAttribute('autoplay');
      obs.observe(v);
    });
  }

  /* ---- Highlight the active section in the table of contents ---- */
  const tocLinks = Array.from(document.querySelectorAll('.toc a'));
  const targets = tocLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  if (targets.length && 'IntersectionObserver' in window) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            const id = e.target.id;
            tocLinks.forEach((a) =>
              a.classList.toggle('active', a.getAttribute('href') === '#' + id)
            );
          }
        });
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );
    targets.forEach((t) => spy.observe(t));
  }
});
