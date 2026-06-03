# PhyGS — Project Page (gh-pages)

Static project page for **PhyGS: Physically-Grounded Controllable Scene Generation**
(University of Michigan). Plain HTML/CSS/JS, no build step.

These files are the **website** and live on the `gh-pages` branch of the
`m-and-m-lab/PhyGS` repo. Your code lives on `main` of the same repo.

Live URL: **https://m-and-m-lab.github.io/PhyGS/**  (case-sensitive — capital P, GS)

```
index.html  styles.css  .nojekyll  README.md
assets/  (js, images, papers/phygs.pdf, qr, ...)   scripts/make_qr.py
```

## Deploy to the gh-pages branch
The whole repo must be **public** for free GitHub Pages. Easiest safe method — build
the page branch in a fresh clone so your code clone is untouched:

```bash
git clone https://github.com/m-and-m-lab/PhyGS.git PhyGS-page
cd PhyGS-page
git checkout --orphan gh-pages      # new branch with no history
git rm -rf .                        # empties the working tree (code stays on main)
cp -r /path/to/phygs-site/. .       # drop these website files in at the root
git add .
git commit -m "PhyGS project page"
git push -u origin gh-pages
```
Then: repo **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `(root)`**.
Live in ~1 min at the URL above. Delete the `PhyGS-page` folder afterwards; `main` is untouched.

## Add figures / videos
Export from the paper into `assets/images/`: `teaser.png` (Fig 1), `pipeline.png`
(Fig 2), `export.png` (Fig 3); add videos to `assets/videos/`. For each, delete the
grey `.placeholder` div and paste the commented `<video>` snippet beside it (files < 100 MB).

## QR (standalone, not on the page)
`python scripts/make_qr.py "https://m-and-m-lab.github.io/PhyGS/"` → `assets/qr/project_qr.png`.
