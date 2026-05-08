# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal project portfolio site for Dayan Parker (EE student at WashU), built with [Jekyll](https://jekyllrb.com) using the [Lanyon](https://github.com/poole/lanyon) theme. Deployed to GitHub Pages at `https://dayan-parker.github.io/dayanparker/`.

## Local development

```bash
jekyll serve          # build and serve at http://localhost:4000
jekyll build          # build to _site/ without serving
```

There is no Gemfile; Jekyll is expected to be installed globally. The `_site/` directory and `.jekyll-cache/` are gitignored and never committed.

## Adding content

**New project post:** create `_posts/YYYY-MM-DD-slug.md` with this front matter:

```markdown
---
layout: post
title: Your Post Title
---
```

Posts appear on the home page in reverse-chronological order (paginated to 5 per page via `jekyll-paginate`).

**New sidebar page:** use `layout: page` — the sidebar auto-discovers all pages with this layout and sorts them by URL.

## Architecture

- `_layouts/default.html` — root layout; sets the theme class (`theme-base-08`) on `<body>`. Change the theme here.
- `_layouts/post.html` / `_layouts/page.html` — extend default; post layout adds date and related-posts links.
- `_includes/sidebar.html` — renders the slide-out nav; dynamically generates links from `layout: page` pages.
- `_includes/head.html` — loads CSS, fonts, and (if configured) Google Analytics.
- `_config.yml` — site title, URL, author, pagination, and plugin config.
- `index.html` — home page; renders paginated posts via Liquid.
- `public/css/` — three CSS files: `poole.css` (base), `lanyon.css` (theme/sidebar), `syntax.css` (code highlighting).

## Images and media

Images and video are committed to `images/`. For any image that may not yet be deployed to GitHub Pages (e.g. newly added files), use Jekyll's `absolute_url` filter so it resolves correctly both locally and in production:

```html
<img src="{{ '/images/your-image.png' | absolute_url }}" ...>
```

Do not use bare relative paths (e.g. `images/foo.png`) — they break on the paginated home page due to URL depth differences. Hardcoded absolute GitHub Pages URLs (`https://dayan-parker.github.io/dayanparker/images/...`) also work for images already deployed, but will 404 locally for newly added files that haven't been pushed yet.

## Math rendering

MathJax is **not** loaded globally. Add the following script tag at the top of any post that uses LaTeX:

```html
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"></script>
```

## Branch notes

Per the upstream Lanyon README: `gh_pages` is the **live hosted branch** (currently the only branch in use). The upstream convention was `master` for development, but this repo works directly on `gh_pages`.
